import chess
import chess.engine
from openai import OpenAI
import random

client = OpenAI()

# Initialize the Stockfish engine
engine = chess.engine.SimpleEngine.popen_uci('./stockfish')  # Update the path to your Stockfish executable

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
    if response_content in ['1-0', '0-1', '1/2-1/2']:
        return response_content  # Game result
    tokens = response_content.split()
    if len(tokens) < 2 or len(tokens) > 5:
        return None
    move_san = tokens[1]  # The move is the second token
    return move_san

def play_game(gpt4o_plays_white):
    board = chess.Board()
    messages = []
    illegal_move_attempts = 0
    max_attempts = 5
    model = 'gpt-4o'  # Replace with the actual model name
    move_number = 1

    # Initialize the conversation
    messages.append({'role': 'system', 'content': 'You are an expert chess player.'})
    if gpt4o_plays_white:
        # GPT-4o plays white and starts the conversation
        messages.append({'role': 'user', 'content': 'Hi! Let\'s play chess. Please make the first move.'})
    else:
        # GPT-4o plays black; no initial message needed
        pass

    # Start the game loop
    while not board.is_game_over():
        if (board.turn == chess.WHITE and gpt4o_plays_white) or (board.turn == chess.BLACK and not gpt4o_plays_white):
            # GPT-4o's turn
            move_attempts = 0
            legal_move_made = False
            while not legal_move_made and move_attempts < max_attempts:
                # Get GPT-4o's move
                response = get_completion(model, messages)
                messages.append({'role': 'assistant', 'content': response['content']})

                # Check if GPT-4o declares the game over
                if response['content'].strip() in ['1-0', '0-1', '1/2-1/2']:
                    # TODO: ADD SOMETHING THAT CHECKS board.is_game_over() and reprompts model if not
                    # Game over
                    # result = response['content'].strip()
                    result = board.result()
                    return result, illegal_move_attempts

                # Parse the move from GPT-4o's response
                move_san = parse_move(response['content'])
                if not move_san:
                    # Invalid format
                    illegal_move_attempts += 1
                    move_attempts += 1
                    messages.append({'role': 'user', 'content': 'Invalid move format. Please provide your move in the format "move_number. move" or "move_number... move".'})
                    # TODO: Don't ask the model, simply reprompt from the previous message
                    continue

                # Check if the move is legal
                try:
                    move = board.parse_san(move_san)
                    board.push(move)
                    legal_move_made = True
                    if not gpt4o_plays_white and board.turn == chess.WHITE:
                        move_number +=1
                except ValueError:
                    # Move is illegal
                    illegal_move_attempts += 1
                    move_attempts += 1
                    # Re-prompt GPT-4o
                    messages.append({'role': 'user', 'content': 'Illegal move. Please make a legal move.'})
            if not legal_move_made:
                # GPT-4o failed to make a legal move after max_attempts
                result = 'Stockfish wins (GPT-4o failed to make a legal move)'
                return result, illegal_move_attempts
            if board.is_game_over():
                break
        else:
            # Stockfish's turn
            stockfish_result = engine.play(board, chess.engine.Limit(depth=10))  # You can adjust the depth
            board.push(stockfish_result.move)
            # Log the move
            move_san = board.san(stockfish_result.move)
            # Construct the move string with move number
            if gpt4o_plays_white:
                # GPT-4o is White, Stockfish is Black
                move_str = f"{move_number}... {move_san}"
                move_number +=1
            else:
                # GPT-4o is Black, Stockfish is White
                move_str = f"{move_number}. {move_san}"
            messages.append({'role': 'user', 'content': move_str})
            if board.is_game_over():
                break

    # Determine the result
    result = board.result()
    return result, illegal_move_attempts

# Main loop to play multiple games
num_games = 10  # Set the number of games you want to play
gpt4o_wins = 0
stockfish_wins = 0
draws = 0
illegal_move_counts = []

for i in range(num_games):
    gpt4o_plays_white = (i % 2 == 0)  # Alternate colors
    result, illegal_moves = play_game(gpt4o_plays_white)
    illegal_move_counts.append(illegal_moves)
    print(f"Game {i+1} result: {result}, Illegal moves: {illegal_moves}")
    if result == '1-0':
        if gpt4o_plays_white:
            gpt4o_wins +=1
        else:
            stockfish_wins +=1
    elif result == '0-1':
        if gpt4o_plays_white:
            stockfish_wins +=1
        else:
            gpt4o_wins +=1
    elif result == '1/2-1/2':
        draws +=1
    elif 'GPT-4o failed to make a legal move' in result:
        stockfish_wins +=1
    else:
        # Handle other results
        pass

# Calculate ELO rating
# Assuming Stockfish's ELO at depth 10 is approximately 2800 (this is a rough estimate)
stockfish_elo = 2800

# Calculate GPT-4o's score
total_games = gpt4o_wins + stockfish_wins + draws
gpt4o_score = gpt4o_wins + 0.5 * draws

# Calculate expected score based on ELO difference
def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

# Since we don't know GPT-4o's ELO, we can approximate it
# Let's assume GPT-4o's ELO is X, and solve for X

# Let's define a function to estimate GPT-4o's ELO
def estimate_elo(gpt4o_score, total_games, stockfish_elo):
    # Initialize GPT-4o's ELO
    gpt4o_elo = 1200  # Starting estimate
    for _ in range(20):  # Iterate to refine the estimate
        exp_score = expected_score(gpt4o_elo, stockfish_elo) * total_games
        # Adjust GPT-4o's ELO based on actual score vs expected score
        gpt4o_elo += (gpt4o_score - exp_score) * 32  # K-factor of 32
    return gpt4o_elo

gpt4o_estimated_elo = estimate_elo(gpt4o_score, total_games, stockfish_elo)
print(f"Estimated ELO of GPT-4o: {gpt4o_estimated_elo}")

# Print the results
print(f"GPT-4o wins: {gpt4o_wins}")
print(f"Stockfish wins: {stockfish_wins}")
print(f"Draws: {draws}")
print(f"Average illegal moves per game: {sum(illegal_move_counts)/len(illegal_move_counts)}")