import chess
import chess.engine
import chess.pgn
from openai import AsyncOpenAI
from datetime import datetime
import asyncio
import aiofiles
from dataclasses import dataclass
from typing import Optional
import logging

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
[TimeControl "120'/60 +, 60'/20m + 15'/end & 30\"/m from move 61"]
[Variant "Standard"]

'''

@dataclass
class GameResult:
    game: chess.pgn.Game
    result: str
    illegal_moves: int
    pgn: str

class BaseChessRunner:
    def __init__(self, model_name: str, stockfish_path: str = './stockfish', engine_elo: int = 1600):
        self.model = model_name
        self.stockfish_path = stockfish_path
        self.engine_elo = engine_elo
        self.client = AsyncOpenAI()
        
    def format_pgn_for_prompt(self, moves: list[chess.Move], board: chess.Board, result: Optional[str] = None) -> str:
        """Format moves into a PGN string for the prompt."""
        pgn = INITIAL_PROMPT.format(result=result if result else "*")
        
        if not moves:
            if board.turn == chess.WHITE:
                return pgn + "\n1."
            return pgn
            
        # Create a temporary board to generate SAN notation
        temp_board = chess.Board()
        move_texts = []
        
        for i in range(0, len(moves), 2):
            move_num = i//2 + 1
            # Generate SAN for white's move using current board state
            white_move = temp_board.san(moves[i])
            temp_board.push(moves[i])
            move_texts.append(f"{move_num}. {white_move}")
            
            if i + 1 < len(moves):  # If there's a black move
                black_move = temp_board.san(moves[i + 1])
                temp_board.push(moves[i + 1])
                move_texts.append(black_move)
                # if i + 2 < len(moves):  # If there are more moves, add the next number
                #     move_texts.append(f"{move_num + 1}.")
            
        pgn += "\n" + " ".join(move_texts)
        
        # Add the current move number if we're waiting for white
        if board.turn == chess.WHITE and board.fullmove_number != 1 and moves:
            pgn += f" {board.fullmove_number}."
            
        return pgn

    def format_move(self, move: str) -> str:
        """Clean up move string from model output."""
        return move.lstrip().split(' ')[0].split('\n')[0]

    async def get_model_move(self, board: chess.Board, pgn: str, max_retries: int = 5) -> Optional[chess.Move]:
        """Get and validate a move from the base model."""
        attempt = 0
        while attempt < max_retries:
            try:
                response = await self.client.completions.create(
                    model=self.model,
                    prompt=pgn,
                    max_tokens=10,
                    temperature=0.7,
                    stop=[".", "\n"]
                )
                
                move_str = self.format_move(response.choices[0].text)
                try:
                    move = board.parse_san(move_str)
                    if move in board.legal_moves:
                        return move
                except ValueError:
                    pass
                
                attempt += 1
                logger.warning(f"Attempt {attempt}: Invalid move '{move_str}'")
                
            except Exception as e:
                logger.error(f"API call failed: {e}")
                attempt += 1
                
        return None

    async def play_game(self, game_number: int, gpt_plays_white: bool) -> GameResult:
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': self.engine_elo})
        
        try:
            board = chess.Board()
            moves = []
            illegal_move_count = 0
            
            # Initialize game
            game = chess.pgn.Game()
            game.headers.update({
                "Event": f"{self.model} vs Stockfish",
                "Site": "Local",
                "Date": datetime.now().strftime("%Y.%m.%d"),
                "Round": str(game_number+1),
                "White": self.model if gpt_plays_white else "Stockfish",
                "Black": "Stockfish" if gpt_plays_white else self.model
            })
            node = game

            # Set expected result based on who plays white
            expected_result = "1-0" if gpt_plays_white else "0-1"
            move_num = 1
            while not board.is_game_over():
                if (board.turn == chess.WHITE and gpt_plays_white) or (board.turn == chess.BLACK and not gpt_plays_white):
                    # GPT's turn
                    pgn = self.format_pgn_for_prompt(moves, board, expected_result)
                    move = await self.get_model_move(board, pgn)
                    
                    if not move:
                        result = 'Failed'
                        game.headers["Result"] = result
                        return GameResult(game, result, illegal_move_count, pgn)
                        
                    logger.info(f"Game {game_number+1} - {self.model} move: {move_num}. {board.san(move)}")
                    
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
    def estimate_elo(gpt_score: float, total_games: int, stockfish_elo: int) -> float:
        def expected_score(elo_a: float, elo_b: float) -> float:
            return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

        gpt_elo = 1200
        for _ in range(20):
            exp_score = expected_score(gpt_elo, stockfish_elo) * total_games
            gpt_elo += ((gpt_score - exp_score) / total_games) * 40
        return gpt_elo

async def main():
    num_games = 100  # Reduced from 150 since base model calls are more expensive
    concurrent_games = 10  # Reduced from 10 to be gentler on API limits
    model_name = 'gpt-4-base'
    
    runner = BaseChessRunner(model_name)
    
    async def write_game(game_result: GameResult):
        async with aiofiles.open('game_logs.pgn', 'a') as pgn_file:
            await pgn_file.write(str(game_result.game) + "\n\n")

    scores = {
        'gpt_wins': 0,
        'stockfish_wins': 0,
        'draws': 0,
        'failed_games': 0,
        'illegal_moves': []
    }

    for batch_start in range(0, num_games, concurrent_games):
        batch_size = min(concurrent_games, num_games - batch_start)
        tasks = []
        
        for i in range(batch_size):
            game_number = batch_start + i
            gpt_plays_white = (game_number % 2 == 0)
            tasks.append(runner.play_game(game_number, gpt_plays_white))
        
        results = await asyncio.gather(*tasks)
        
        for game_result in results:
            await write_game(game_result)
            scores['illegal_moves'].append(game_result.illegal_moves)
            
            if game_result.result == '1-0':
                if game_result.game.headers['White'] == model_name:
                    scores['gpt_wins'] += 1
                else:
                    scores['stockfish_wins'] += 1
            elif game_result.result == '0-1':
                if game_result.game.headers['Black'] == model_name:
                    scores['gpt_wins'] += 1
                else:
                    scores['stockfish_wins'] += 1
            elif game_result.result == '1/2-1/2':
                scores['draws'] += 1
            elif game_result.result == 'Failed':
                scores['failed_games'] += 1

    total_games = scores['gpt_wins'] + scores['stockfish_wins'] + scores['draws']
    gpt_score = scores['gpt_wins'] + 0.5 * scores['draws']
    
    if total_games > 0:
        gpt_estimated_elo = runner.estimate_elo(gpt_score, total_games, runner.engine_elo)
        logger.info(f"Total successful games: {total_games}")
        logger.info(f"Estimated Elo for {model_name}: {gpt_estimated_elo:.2f}")

        # Write final results
        async with aiofiles.open('game_scores.txt', 'a') as f:
            await f.write("\n###\n")
            await f.write(f"Estimated Elo score for {model_name}: {gpt_estimated_elo:.2f}\n")
            await f.write(f"Model wins: {scores['gpt_wins']}\n")
            await f.write(f"Stockfish wins: {scores['stockfish_wins']}\n")
            await f.write(f"Draws: {scores['draws']}\n")
            await f.write(f"Failed games: {scores['failed_games']}\n")
            await f.write(f"Average illegal moves per game: {sum(scores['illegal_moves'])/len(scores['illegal_moves']):.2f}\n")
            await f.write("###\n")
    else:
        logger.info(f"All {scores['failed_games']} games failed.")

if __name__ == "__main__":
    asyncio.run(main())