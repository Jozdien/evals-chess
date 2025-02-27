import chess
import chess.pgn
import chess.engine
from stockfish import Stockfish
import io
import re
import statistics
from datetime import datetime
from collections import defaultdict

def extract_stats_from_logs(input_file, output_file):
    """
    Extracts statistics blocks from PGN files and writes them to an output file.
    Stats blocks are identified by starting with '###' and containing specific metrics.
    """
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split the content by '###' and filter for blocks that look like stats
    blocks = content.split('###')
    stats_blocks = []
    
    for block in blocks:
        # Skip empty blocks
        if not block.strip():
            continue
            
        # Check if this block contains the key metrics we're looking for
        if all(indicator in block.lower() for indicator in ['elo', 'wins', 'draws']):
            stats_blocks.append(f"###\n{block.strip()}\n###\n")
    
    # Write the extracted blocks to the output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(stats_blocks))

def expected_score(elo_a, elo_b):
    """Calculate expected score based on ELO difference."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def estimate_elo(actual_score, total_games, opponent_elo):
    """Estimate ELO rating based on performance against an opponent."""
    # Initialize ELO
    elo = 1200  # Starting estimate
    for _ in range(20):  # Iterate to refine the estimate
        exp_score = expected_score(elo, opponent_elo) * total_games
        # Adjust ELO based on actual score vs expected score
        elo += ((actual_score - exp_score) / total_games) * 40  # K-factor of 40
    return elo

def analyze_pgn_file(file_path, stockfish_elo=1600):
    """Analyzes a PGN file and extracts statistics for each model."""
    
    # Dictionary to store stats for each model
    model_stats = defaultdict(lambda: {
        'games': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'failed_games': 0,
        'illegal_moves': [],
        'as_white': 0,
        'as_black': 0
    })
    
    current_model = None
    
    with open(file_path) as pgn_file:
        content = pgn_file.read()
        
        # Split content into sections (each game and stats block)
        sections = content.split('\n\n')
        
        i = 0
        while i < len(sections):
            section = sections[i]
            
            # Check if this section is a stats block
            if section.strip().startswith('###'):
                # Extract model name and stats from the block
                stats_match = re.search(r'Estimated Elo score for (.*?): ([\d.]+)', section)
                if stats_match:
                    model_name = stats_match.group(1)
                    current_model = model_name
            
            # Check if this section starts a game
            elif '[Event' in section:
                game_text = section
                # Combine with next section if it's the moves
                if i + 1 < len(sections) and not sections[i + 1].strip().startswith('['):
                    game_text += '\n\n' + sections[i + 1]
                    i += 1
                
                # Parse game
                pgn = chess.pgn.read_game(io.StringIO(game_text))
                if pgn:
                    headers = pgn.headers
                    model_name = None
                    
                    # Extract model name from players
                    if 'gpt' in headers['White']:
                        model_name = headers['White']
                        model_stats[model_name]['as_white'] += 1
                    elif 'gpt' in headers['Black']:
                        model_name = headers['Black']
                        model_stats[model_name]['as_black'] += 1
                    
                    if model_name:
                        # Update game count
                        model_stats[model_name]['games'] += 1
                        
                        # Update illegal moves
                        illegal_moves = int(headers.get('IllegalMoves', '0'))
                        model_stats[model_name]['illegal_moves'].append(illegal_moves)
                        
                        # Update game results
                        result = headers['Result']
                        if result == '1-0':
                            if headers['White'] == model_name:
                                model_stats[model_name]['wins'] += 1
                            else:
                                model_stats[model_name]['losses'] += 1
                        elif result == '0-1':
                            if headers['Black'] == model_name:
                                model_stats[model_name]['wins'] += 1
                            else:
                                model_stats[model_name]['losses'] += 1
                        elif result == '1/2-1/2':
                            model_stats[model_name]['draws'] += 1
                        elif result == 'Failed':
                            model_stats[model_name]['failed_games'] += 1
                            model_stats[model_name]['losses'] += 1  # Count failed games as losses
            
            i += 1
    
    # Calculate and print statistics for each model
    for model_name, stats in model_stats.items():
        total_illegal_moves = sum(stats['illegal_moves'])
        avg_illegal_moves = total_illegal_moves / len(stats['illegal_moves']) if stats['illegal_moves'] else 0
        
        print(f"\nStatistics for model: {model_name}")
        print(f"Total games: {stats['games']}")
        print(f"Games as White: {stats['as_white']}")
        print(f"Games as Black: {stats['as_black']}")
        print(f"Wins: {stats['wins']}")
        print(f"Losses: {stats['losses']}")  # This now includes failed games
        print(f"Draws: {stats['draws']}")
        print(f"Failed games: {stats['failed_games']}")
        print(f"Average illegal moves per game: {avg_illegal_moves:.2f}")
        
        # Calculate win rates and ELO - both including and excluding failed games
        completed_games_with_fails = stats['wins'] + stats['losses'] + stats['draws']  # Failed games counted in losses
        completed_games_no_fails = completed_games_with_fails - stats['failed_games']  # Excluding failed games
        
        if completed_games_with_fails > 0:
            # Stats including failed games as losses
            win_rate_with_fails = (stats['wins'] + 0.5 * stats['draws']) / completed_games_with_fails
            print(f"Win rate (counting fails as losses): {win_rate_with_fails:.2%}")
            
            actual_score_with_fails = stats['wins'] + 0.5 * stats['draws']
            estimated_elo_with_fails = estimate_elo(actual_score_with_fails, completed_games_with_fails, stockfish_elo)
            print(f"Estimated ELO rating (counting fails as losses): {estimated_elo_with_fails:.2f}")
        
        if completed_games_no_fails > 0:
            # Stats excluding failed games
            wins_no_fails = stats['wins']
            losses_no_fails = stats['losses'] - stats['failed_games']
            win_rate_no_fails = (wins_no_fails + 0.5 * stats['draws']) / completed_games_no_fails
            print(f"Win rate (excluding failed games): {win_rate_no_fails:.2%}")
            
            actual_score_no_fails = wins_no_fails + 0.5 * stats['draws']
            estimated_elo_no_fails = estimate_elo(actual_score_no_fails, completed_games_no_fails, stockfish_elo)
            print(f"Estimated ELO rating (excluding failed games): {estimated_elo_no_fails:.2f}")

def count_games(file_path):
    """Count the total number of games in a PGN file."""
    game_count = 0
    with open(file_path) as pgn_file:
        content = pgn_file.read()
        # Count occurrences of [Event tags, which indicate the start of a new game
        game_count = content.count('[Event "')
    return game_count

def evaluate_position(fen_string, depth=20):
    """
    Evaluates a chess position using Stockfish.
    
    Args:
        fen_string (str): The FEN notation of the chess position
        depth (int): Search depth for Stockfish (default=20)
    
    Returns:
        float: Evaluation score in centipawns (positive for white advantage)
    """
    try:
        # Initialize Stockfish (you'll need to specify your path)
        stockfish = Stockfish(path="./stockfish")
        
        # Set position from FEN
        stockfish.set_fen_position(fen_string)
        
        # Set analysis depth
        stockfish.set_depth(depth)
        
        # Get evaluation
        evaluation = stockfish.get_evaluation()
        
        # Convert mate scores to high numerical values
        if evaluation.get('type') == 'mate':
            score = 10000 if evaluation['value'] > 0 else -10000
        else:
            score = evaluation['value']
            
        return score
        
    except Exception as e:
        raise Exception(f"Evaluation failed: {str(e)}")

def extract_failed_games(input_file, output_file, evaluation_depth=20):
    """
    [Previous docstring remains the same]
    """
    try:
        with open(input_file, 'r') as f:
            content = f.read()
        
        games = content.split('\n\n[Event')
        if len(games) > 1:
            games[1:] = ['[Event' + game for game in games[1:]]
        
        failed_games = []
        for game in games:
            if '[Result "Failed"]' in game:
                game_io = io.StringIO(game)
                pgn_game = chess.pgn.read_game(game_io)
                if pgn_game is None:
                    continue
                
                board = pgn_game.board()
                for move in pgn_game.mainline_moves():
                    board.push(move)
                
                eval_score = evaluate_position(board.fen(), evaluation_depth)
                eval_str = f"{eval_score/100:.2f}" if abs(eval_score) < 10000 else "M" + str(eval_score//10000)
                eval_header = f'[StockfishEval "{eval_str}"]'
                
                # Split the game into lines
                lines = game.split('\n')
                
                # Find the index of the IllegalMoves header
                illegal_moves_idx = -1
                for i, line in enumerate(lines):
                    if line.startswith('[IllegalMoves'):
                        illegal_moves_idx = i
                        break
                
                if illegal_moves_idx != -1:
                    # Insert the evaluation header after IllegalMoves
                    lines.insert(illegal_moves_idx + 1, eval_header)
                else:
                    # Fallback: insert before the moves if IllegalMoves header not found
                    move_start_idx = next((i for i, line in enumerate(lines) if line.startswith('1. ')), len(lines))
                    lines.insert(move_start_idx, eval_header)
                
                processed_game = '\n'.join(lines)
                failed_games.append(processed_game)
        
        with open(output_file, 'w') as f:
            f.write('\n\n\n'.join(failed_games))
        
        return len(failed_games)
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return 0
    
def analyze_failed_games(file_path):
    with open(file_path) as pgn_file:
        pgn_text = pgn_file.read()

    games = pgn_text.strip().split('\n\n\n')
    model_stats = defaultdict(lambda: {'evals': [], 'winning_states': 0, 'losing_states': 0})
    
    for game in games:
        white = re.search(r'\[White "(.+?)"\]', game).group(1)
        black = re.search(r'\[Black "(.+?)"\]', game).group(1)
        eval_str = re.search(r'\[StockfishEval "(.+?)"\]', game).group(1)
        
        # Handle mate scores
        if eval_str.startswith('M'):
            mate_in = int(eval_str[1:])
            if mate_in > 0:  # Positive mate score = white is winning
                model_stats[white]['winning_states'] += 1
                model_stats[black]['losing_states'] += 1
            else:
                model_stats[white]['losing_states'] += 1
                model_stats[black]['winning_states'] += 1
            continue
            
        # Regular eval scores
        eval_score = float(eval_str)
        model_stats[white]['evals'].append(eval_score)
        model_stats[black]['evals'].append(-eval_score)
    
    # Calculate stats
    results = {}
    for model, stats in model_stats.items():
        if not stats['evals'] and not (stats['winning_states'] or stats['losing_states']):
            continue
        evals = stats['evals']
        results[model] = {
            'avg_eval': statistics.mean(evals) if evals else None,
            'std_eval': statistics.stdev(evals) if len(evals) > 1 else None,
            'quartiles': statistics.quantiles(evals) if len(evals) >= 4 else None,
            'winning_states': stats['winning_states'],
            'losing_states': stats['losing_states']
        }
    
    for model, stats in results.items():
        print(f"\n{model}:")
        if stats['avg_eval'] is not None:
            std_str = f" (±{stats['std_eval']:.2f})" if stats['std_eval'] is not None else ""
            print(f"Average eval: {stats['avg_eval']:.2f}{std_str}")
            if stats['quartiles']:
                print(f"Quartiles: {[f'{q:.2f}' for q in stats['quartiles']]}")
        print(f"Winning positions: {stats['winning_states']}")
        print(f"Losing positions: {stats['losing_states']}")

if __name__ == "__main__":
    # print(f"Total number of games: {count_games('game_logs.pgn')}")
    # extract_failed_games('game_logs.pgn', 'failed_game_logs.pgn', evaluation_depth=20)
    analyze_pgn_file('game_logs.pgn')
    # analyze_failed_games('failed_game_logs.pgn')