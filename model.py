import chess
import chess.engine
import chess.pgn
from openai import OpenAI
from datetime import datetime
import random

client = OpenAI()
model = 'gpt-4-0314'  # Replace with the actual model name

# Initialize the Stockfish engine
engine = chess.engine.SimpleEngine.popen_uci('./stockfish')  # Update the path to your Stockfish executable
engine_elo = 1600
engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': engine_elo})

def get_completion(model, messages):
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message

def parse_move(response_content):
    """
    Parses the move from GPT-4o's response content.
    Expects the move in the format like '1. e4' or '1... c5'
    Returns the move in SAN notation, e.g., 'e4' or 'c5'
    """
    response_content = response_content.strip()
    tokens = response_content.split()
    if len(tokens) != 2:
        return None
    move_san = tokens[1]  # The move is the second token
    return move_san

def play_game(gpt4o_plays_white, game_number):
    board = chess.Board()
    messages = []
    illegal_move_attempts = 0
    max_attempts = 5
    move_number = 1

    # Initialize the conversation
    messages.append({'role': 'system', 'content': 'You are an expert chess player. Please ONLY reply with your move response in full PGN notation (with the move number, etc), without any further text.'})
    if gpt4o_plays_white:
        # GPT-4o plays white and starts the conversation
        messages.append({'role': 'user', 'content': 'Hi! Let\'s play chess. Please make the first move.'})
    else:
        # GPT-4o plays black; opponent starts
        pass

    print("GAME STARTING")

    # Create a Game object for PGN
    game = chess.pgn.Game()
    game.headers["Event"] = f"{model} vs Stockfish"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_number)
    game.headers["White"] = model if gpt4o_plays_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if gpt4o_plays_white else model
    node = game  # Initialize the node to build moves

    # Start the game loop
    while not board.is_game_over():
        if (board.turn == chess.WHITE and gpt4o_plays_white) or (board.turn == chess.BLACK and not gpt4o_plays_white):
            # GPT-4o's turn
            move_attempts = 0
            legal_move_made = False
            while not legal_move_made and move_attempts < max_attempts:
                # Get GPT-4o's move
                response = get_completion(model, messages)
                response_content = response.content.strip()
                messages.append({'role': 'assistant', 'content': response_content})

                # Parse the move from GPT-4o's response
                move_san = parse_move(response_content)

                # Check if GPT-4o declared the game over incorrectly
                if not move_san:
                    if board.is_game_over():
                        print("GAME OVER")
                        # Game is actually over
                        break
                    else:
                        # Game is not over, re-prompt
                        illegal_move_attempts += 1
                        move_attempts += 1
                        # Remove last assistant message
                        messages.pop()

                        print("ILLEGAL MOVE MADE")

                        continue

                # Check if the move is legal
                try:
                    move = board.parse_san(move_san)
                    board.push(move)
                    node = node.add_variation(move)  # Add move to PGN
                    legal_move_made = True

                    # Construct the move string with move number
                    if board.turn == chess.WHITE:
                        # GPT-4o just played Black
                        move_str = f"{move_number}... {move_san}"
                        move_number += 1  # Increment after Black's move
                    else:
                        # GPT-4o just played White
                        move_str = f"{move_number}. {move_san}"

                    print(f"MOVE MADE BY GPT-4o: {move_str}")
                    # No need to send the move back to GPT-4o, as per your protocol
                except ValueError:
                    # Move is illegal
                    illegal_move_attempts += 1
                    move_attempts += 1
                    # Remove last assistant message
                    messages.pop()

                    print("ILLEGAL MOVE MADE")

            if not legal_move_made:
                # GPT-4o failed to make a legal move after max_attempts
                result = 'Failed'
                game.headers["Result"] = result
                game.headers["IllegalMoves"] = str(illegal_move_attempts)
                return game, result, illegal_move_attempts
            if board.is_game_over():
                break
        else:
            # Stockfish's turn
            stockfish_result = engine.play(board, chess.engine.Limit(depth=10))  # Adjust depth as needed
            move_san = board.san(stockfish_result.move)
            board.push(stockfish_result.move)
            node = node.add_variation(stockfish_result.move)  # Add move to PGN
            # Construct the move string with move number
            if board.turn == chess.WHITE:
                # Stockfish just played Black
                move_str = f"{move_number}... {move_san}"
                move_number += 1  # Increment after Black's move
            else:
                # Stockfish just played White
                move_str = f"{move_number}. {move_san}"
            # Send the move to GPT-4o
            print(f"MOVE MADE BY STOCKFISH: {move_str}")
            messages.append({'role': 'user', 'content': move_str})
            if board.is_game_over():
                break

    # Determine the result
    result = board.result()
    game.headers["Result"] = result
    game.headers["IllegalMoves"] = str(illegal_move_attempts)
    return game, result, illegal_move_attempts

# Main loop to play multiple games
num_games = 100  # Set the number of games you want to play
gpt4o_wins = 0
stockfish_wins = 0
draws = 0
failed_games = 0  # Counter for games where GPT-4o failed to make a legal move
illegal_move_counts = []

for i in range(num_games):
    gpt4o_plays_white = (i % 2 == 0)  # Alternate colors
    game, result, illegal_moves = play_game(gpt4o_plays_white, i + 1)
    illegal_move_counts.append(illegal_moves)
    print(f"Game {i+1} result: {result}, Illegal moves: {illegal_moves}")

    # Write the game to a PGN file
    with open('game_logs.pgn', 'a') as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)

    if result == '1-0':
        if gpt4o_plays_white:
            gpt4o_wins += 1
        else:
            stockfish_wins += 1
    elif result == '0-1':
        if gpt4o_plays_white:
            stockfish_wins += 1
        else:
            gpt4o_wins += 1
    elif result == '1/2-1/2':
        draws += 1
    elif result == 'Failed':
        failed_games += 1
    else:
        # Handle other results if any
        pass

# Calculate ELO rating
# Use the configured Stockfish ELO
stockfish_elo = engine_elo

# Calculate GPT-4o's score
total_games = gpt4o_wins + stockfish_wins + draws  # Exclude failed games
gpt4o_score = gpt4o_wins + 0.5 * draws

# Calculate expected score based on ELO difference
def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

# Since we don't know GPT-4o's ELO, we can approximate it
def estimate_elo(gpt4o_score, total_games, stockfish_elo):
    # Initialize GPT-4o's ELO
    gpt4o_elo = 1200  # Starting estimate
    for _ in range(20):  # Iterate to refine the estimate
        exp_score = expected_score(gpt4o_elo, stockfish_elo) * total_games
        # Adjust GPT-4o's ELO based on actual score vs expected score
        gpt4o_elo += ((gpt4o_score - exp_score) / total_games) * 40  # K-factor of 40
    return gpt4o_elo

if total_games > 0:
    gpt4o_estimated_elo = estimate_elo(gpt4o_score, total_games, stockfish_elo)
    print(f"Estimated Elo of GPT-4o: {gpt4o_estimated_elo:.2f}")
else:
    print("No completed games to estimate ELO.")

# Print the results
print(f"GPT-4o wins: {gpt4o_wins}")
print(f"Stockfish wins: {stockfish_wins}")
print(f"Draws: {draws}")
print(f"Failed games (GPT-4o couldn't make a legal move): {failed_games}")
print(f"Average illegal moves per game: {sum(illegal_move_counts)/len(illegal_move_counts):.2f}")

with open('game_logs.pgn', 'a') as pgn_file:
    pgn_file.write("\n###\n")
    pgn_file.write(f"Estimated Elo score for {model}: {gpt4o_estimated_elo:.2f}\n")
    pgn_file.write(f"Model wins: {gpt4o_wins}\n")
    pgn_file.write(f"Stockfish wins: {stockfish_wins}\n")
    pgn_file.write(f"Draws: {draws}\n")
    pgn_file.write(f"Failed games (GPT-4o couldn't make a legal move): {failed_games}\n")
    pgn_file.write(f"Average illegal moves per game: {sum(illegal_move_counts)/len(illegal_move_counts):.2f}")
    pgn_file.write("\n###\n")

# Close the Stockfish engine
engine.quit()