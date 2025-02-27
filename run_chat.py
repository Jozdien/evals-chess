import chess
import chess.engine
import chess.pgn
from openai import AsyncOpenAI
from datetime import datetime
import asyncio
import aiofiles
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GameResult:
    game: chess.pgn.Game
    result: str
    illegal_moves: int

class AsyncChessRunner:
    def __init__(self, model_name: str, stockfish_path: str = './stockfish', engine_elo: int = 1600):
        self.model = model_name
        self.stockfish_path = stockfish_path
        self.engine_elo = engine_elo
        self.client = AsyncOpenAI()

    async def get_completion(self, messages):
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return completion.choices[0].message
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

    def parse_move(self, response_content: str) -> Optional[str]:
        response_content = response_content.strip()
        tokens = response_content.split()
        if len(tokens) > 3:
            return None
        return tokens[-1]

    async def play_game(self, game_number: int, gpt_plays_white: bool) -> GameResult:
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': self.engine_elo})
        
        try:
            board = chess.Board()
            messages = []
            illegal_move_attempts = 0
            max_attempts = 5
            move_number = 1

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

            messages.append({
                'role': 'system',
                'content': 'You are an expert chess player. Please ONLY reply with your move response in full PGN notation (with the move number, etc), without any further text.'
            })
            
            if gpt_plays_white:
                messages.append({
                    'role': 'user',
                    'content': 'Hi! Let\'s play chess. Please make the first move.'
                })

            while not board.is_game_over():
                if (board.turn == chess.WHITE and gpt_plays_white) or (board.turn == chess.BLACK and not gpt_plays_white):
                    # GPT's turn
                    move_attempts = 0
                    legal_move_made = False
                    
                    while not legal_move_made and move_attempts < max_attempts:
                        response = await self.get_completion(messages)
                        response_content = response.content.strip()
                        messages.append({'role': 'assistant', 'content': response_content})

                        move_san = self.parse_move(response_content)

                        if not move_san:
                            if board.is_game_over():
                                break
                            illegal_move_attempts += 1
                            move_attempts += 1
                            messages.pop()
                            logger.warning(f"Model returned move in wrong format in game {game_number+1}. Model response: {response_content}")
                            continue

                        try:
                            move = board.parse_san(move_san)
                            board.push(move)
                            node = node.add_variation(move)
                            legal_move_made = True

                            move_str = (f"{move_number}... {move_san}" if board.turn == chess.WHITE 
                                      else f"{move_number}. {move_san}")
                            if board.turn == chess.WHITE:
                                move_number += 1

                            logger.info(f"Game {game_number+1} - {self.model} move: {move_str}")
                        except ValueError:
                            illegal_move_attempts += 1
                            move_attempts += 1
                            messages.pop()
                            logger.warning(f"Game {game_number+1} - Illegal move attempted: {move_san}")

                    if not legal_move_made:
                        result = 'Failed'
                        game.headers.update({"Result": result, "IllegalMoves": str(illegal_move_attempts)})
                        return GameResult(game, result, illegal_move_attempts)
                else:
                    # Stockfish's turn
                    result = engine.play(board, chess.engine.Limit(depth=10))
                    move_san = board.san(result.move)
                    board.push(result.move)
                    node = node.add_variation(result.move)

                    move_str = (f"{move_number}... {move_san}" if board.turn == chess.WHITE 
                              else f"{move_number}. {move_san}")
                    if board.turn == chess.WHITE:
                        move_number += 1

                    logger.info(f"Game {game_number+1} - Stockfish move: {move_str}")
                    messages.append({'role': 'user', 'content': move_str})

                if board.is_game_over():
                    break

            result = board.result()
            game.headers.update({"Result": result, "IllegalMoves": str(illegal_move_attempts)})
            return GameResult(game, result, illegal_move_attempts)

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
    num_games = 150
    concurrent_games = 10  # Adjust based on your API rate limits and system capabilities
    model_name = 'ft:gpt-4o-2024-08-06:personal::AO5x8aoa'
    
    runner = AsyncChessRunner(model_name)
    
    def write_game(game: chess.pgn.Game):
        with open('game_logs.pgn', 'a') as pgn_file:
            exporter = chess.pgn.FileExporter(pgn_file)
            game.accept(exporter)

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
            write_game(game_result.game)
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
    else:  # edge case where all games fail
        logger.info(f"All {scores['failed_games']} games failed.")

if __name__ == "__main__":
    asyncio.run(main())