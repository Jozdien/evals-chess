import chess
import chess.engine
import chess.pgn
from datetime import datetime
import asyncio
import aiofiles
from dataclasses import dataclass
from typing import Optional
import logging
import transformers
import torch
from time import perf_counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INITIAL_PROMPT = '''[Event "FIDE Candidates Tournament 2022"]
[Site "Madrid, Spain"]
[Date "2022.06.17"]
[Round "1.3"]
[White "Caruana, Fabiano"]
[Black "Nakamura, Hikaru"]
[Result "{result}"]
[WhiteElo "2783"]
[WhiteTitle "GM"]
[BlackElo "2760"]
[BlackTitle "GM"]
[Variant "Standard"]

'''

@dataclass
class GameResult:
    game: chess.pgn.Game
    result: str
    illegal_moves: int
    pgn: str
    failed_position_fen: Optional[str] = None  # Store FEN of failed position
    move_history: Optional[str] = None  # Store history of moves up to failure

class LocalChessRunner:
    def __init__(self, model_path: str, stockfish_path: str = './stockfish', engine_elo: int = 1320):
        self.model_path = model_path
        self.stockfish_path = stockfish_path
        self.engine_elo = engine_elo
        
        # Initialize the model pipeline
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
            },
            device_map="auto"
        )
        
    def format_pgn_for_prompt(self, moves: list[chess.Move], board: chess.Board, result: Optional[str] = None) -> str:
        """Format moves into a PGN string for the prompt."""
        pgn = INITIAL_PROMPT.format(result=result if result else "*")
        
        if not moves:
            if board.turn == chess.WHITE:
                return pgn + "\n1."
            return pgn
            
        temp_board = chess.Board()
        move_texts = []
        
        for i in range(0, len(moves), 2):
            move_num = i//2 + 1
            white_move = temp_board.san(moves[i])
            temp_board.push(moves[i])
            move_texts.append(f"{move_num}. {white_move}")
            
            if i + 1 < len(moves):
                black_move = temp_board.san(moves[i + 1])
                temp_board.push(moves[i + 1])
                move_texts.append(black_move)
            
        pgn += "\n" + " ".join(move_texts)
        
        if board.turn == chess.WHITE and board.fullmove_number != 1 and moves:
            pgn += f" {board.fullmove_number}."
            
        return pgn

    def format_move(self, move: str) -> str:
        """Clean up move string from model output."""
        return move.lstrip().split(' ')[0].split('\n')[0]

    async def get_model_move(self, board: chess.Board, pgn: str, max_retries: int = 5) -> Optional[chess.Move]:
        """Get and validate a move from the local model."""
        attempt = 0
        while attempt < max_retries:
            try:
                # Generate move using the local model
                output = self.pipeline(
                    pgn,
                    max_new_tokens=10,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )[0]['generated_text']
                
                # Extract the move from the generated text
                move_str = self.format_move(output[len(pgn):])
                try:
                    move = board.parse_san(move_str)
                    if move in board.legal_moves:
                        return move
                except ValueError:
                    pass
                
                attempt += 1
                logger.warning(f"Attempt {attempt}: Invalid move '{move_str}' at position {board.fen()}")
                
            except Exception as e:
                logger.error(f"Model inference failed: {e}")
                attempt += 1
        
        return None

    async def play_game(self, game_number: int, model_plays_white: bool) -> GameResult:
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': self.engine_elo})
        
        try:
            board = chess.Board()
            moves = []
            illegal_move_count = 0
            
            game = chess.pgn.Game()
            game.headers.update({
                "Event": f"{self.model_path} vs Stockfish",
                "Site": "Local",
                "Date": datetime.now().strftime("%Y.%m.%d"),
                "Round": str(game_number+1),
                "White": self.model_path if model_plays_white else "Stockfish",
                "Black": "Stockfish" if model_plays_white else self.model_path
            })
            node = game

            expected_result = "1-0" if model_plays_white else "0-1"
            move_num = 1
            while not board.is_game_over():
                if (board.turn == chess.WHITE and model_plays_white) or (board.turn == chess.BLACK and not model_plays_white):
                    # Local model's turn
                    pgn = self.format_pgn_for_prompt(moves, board, expected_result)
                    move = await self.get_model_move(board, pgn)
                    
                    if not move:
                        result = 'Failed'
                        game.headers["Result"] = result
                        failed_fen = board.fen()
                        
                        # Format move history
                        move_history = []
                        temp_board = chess.Board()
                        for move in moves:
                            san_move = temp_board.san(move)
                            if temp_board.turn == chess.WHITE:
                                move_history.append(f"{temp_board.fullmove_number}. {san_move}")
                            else:
                                move_history.append(san_move)
                            temp_board.push(move)
                        move_history_str = " ".join(move_history)
                        
                        logger.error(f"Game {game_number+1} failed at position: {failed_fen}")
                        logger.error(f"Move history: {move_history_str}")
                        return GameResult(game, result, illegal_move_count, pgn, failed_fen, move_history_str)
                        
                    logger.info(f"Game {game_number+1} - Model move: {move_num}. {board.san(move)}")
                    
                else:
                    # Stockfish's turn
                    result = engine.play(board, chess.engine.Limit(depth=10))
                    move = result.move
                    logger.info(f"Game {game_number+1} - Stockfish move: {move_num}. {board.san(move)}")
                
                board.push(move)
                moves.append(move)
                node = node.add_variation(move)
                move_num += 1

            result = board.result()
            game.headers["Result"] = result
            final_pgn = self.format_pgn_for_prompt(moves, board, result)
            return GameResult(game, result, illegal_move_count, final_pgn)

        finally:
            engine.quit()

    @staticmethod
    def estimate_elo(model_score: float, total_games: int, stockfish_elo: int) -> float:
        def expected_score(elo_a: float, elo_b: float) -> float:
            return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

        model_elo = 1200
        for _ in range(20):
            exp_score = expected_score(model_elo, stockfish_elo) * total_games
            model_elo += ((model_score - exp_score) / total_games) * 40
        return model_elo

async def main():
    num_games = 1
    concurrent_games = 1
    # model_path = "meta-llama/Llama-3.1-70B"
    model_path = "Qwen/Qwen2.5-72B"
    
    runner = LocalChessRunner(model_path)
    
    async def write_game(game_result: GameResult):
        async with aiofiles.open('game_logs.pgn', 'a') as pgn_file:
            await pgn_file.write(str(game_result.game) + "\n\n")
            if game_result.failed_position_fen:
                await pgn_file.write(f"# Failed at position: {game_result.failed_position_fen}\n")
                await pgn_file.write(f"# Move history: {game_result.move_history}\n\n")

    scores = {
        'model_wins': 0,
        'stockfish_wins': 0,
        'draws': 0,
        'failed_games': 0,
        'illegal_moves': [],
        'failed_positions': []  # New list to track FEN strings of failed positions
    }

    for batch_start in range(0, num_games, concurrent_games):
        batch_size = min(concurrent_games, num_games - batch_start)
        tasks = []
        
        for i in range(batch_size):
            game_number = batch_start + i
            model_plays_white = (game_number % 2 == 0)
            tasks.append(runner.play_game(game_number, model_plays_white))
        
        results = await asyncio.gather(*tasks)
        
        for game_result in results:
            await write_game(game_result)
            scores['illegal_moves'].append(game_result.illegal_moves)
            
            if game_result.failed_position_fen:
                scores['failed_positions'].append(game_result.failed_position_fen)
            
            if game_result.result == '1-0':
                if game_result.game.headers['White'] == model_path:
                    scores['model_wins'] += 1
                else:
                    scores['stockfish_wins'] += 1
            elif game_result.result == '0-1':
                if game_result.game.headers['Black'] == model_path:
                    scores['model_wins'] += 1
                else:
                    scores['stockfish_wins'] += 1
            elif game_result.result == '1/2-1/2':
                scores['draws'] += 1
            elif game_result.result == 'Failed':
                scores['failed_games'] += 1

    total_games = scores['model_wins'] + scores['stockfish_wins'] + scores['draws']
    model_score = scores['model_wins'] + 0.5 * scores['draws']
    
    if total_games > 0:
        model_estimated_elo = runner.estimate_elo(model_score, total_games, runner.engine_elo)
        logger.info(f"Total successful games: {total_games}")
        logger.info(f"Estimated Elo for {model_path}: {model_estimated_elo:.2f}")

        async with aiofiles.open('game_scores.txt', 'a') as f:
            await f.write("\n###\n")
            await f.write(f"Estimated Elo score for {model_path}: {model_estimated_elo:.2f}\n")
            await f.write(f"Model wins: {scores['model_wins']}\n")
            await f.write(f"Stockfish wins: {scores['stockfish_wins']}\n")
            await f.write(f"Draws: {scores['draws']}\n")
            await f.write(f"Failed games: {scores['failed_games']}\n")
            await f.write(f"Average illegal moves per game: {sum(scores['illegal_moves'])/len(scores['illegal_moves']):.2f}\n")
            if scores['failed_positions']:
                await f.write("\nFailed positions:\n")
                for i, game_result in enumerate(results):
                    if game_result.failed_position_fen:
                        await f.write(f"Game {i+1}:\n")
                        await f.write(f"Position: {game_result.failed_position_fen}\n")
                        await f.write(f"Moves: {game_result.move_history}\n\n")
            await f.write("###\n")
    else:
        logger.info(f"All {scores['failed_games']} games failed.")

if __name__ == "__main__":
    asyncio.run(main())