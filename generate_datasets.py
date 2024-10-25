import chess.pgn
import json
import random

# pgn_file = open('chess_db_elo_1400_1600.pgn', 'r')
pgn_file = open('chess_db_elo_2400_2600.pgn', 'r')
# pgn_file = open('chess_db_elo_2400_2600.pgn', 'r')

with open('chess_db_elo_2400_2600.jsonl', 'w') as outfile:
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break

        acceptable_results = {"1-0", "0-1", "1/2-1/2"}
        result = game.headers.get("Result", "*")
        if result not in acceptable_results:
            continue

        time_control = game.headers.get("TimeControl", "")
        if time_control == "60+0":
            continue

        termination = game.headers.get("Termination", "").lower()
        if "abandoned" in termination or "unrated" in termination:
            continue

        messages = []
        board = game.board()
        move_number = 1
        move_count = 0

        assistant_plays_white = random.choice([True, False])

        if assistant_plays_white:
            # The assistant will play White; the user prompts the assistant to start
            messages.append({
                "role": "user",
                "content": "Hi! Let's play chess. Please make the first move."
            })
        else:
            # The assistant will play Black; the user starts the game
            # No need for an extra message since the user will provide the first move below
            pass

        for move in game.mainline_moves():
            san_move = board.san(move)
            if board.turn == assistant_plays_white:
                # It's the assistant's turn
                if board.turn:  # Assistant is White
                    messages.append({
                        "role": "assistant",
                        "content": f"{move_number}. {san_move}"
                    })
                else:  # Assistant is Black
                    messages.append({
                        "role": "assistant",
                        "content": f"{move_number}... {san_move}"
                    })
                    move_number += 1  # Increment after Black's move
            else:
                # It's the user's turn
                if board.turn:  # User is White
                    messages.append({
                        "role": "user",
                        "content": f"{move_number}. {san_move}"
                    })
                else:  # User is Black
                    messages.append({
                        "role": "user",
                        "content": f"{move_number}... {san_move}"
                    })
                    move_number += 1  # Increment after Black's move
            board.push(move)
            move_count += 1

        if move_count < 10:
            continue

        json.dump({"messages": messages}, outfile)
        outfile.write('\n')

pgn_file.close()