import time
import chess.pgn

NUM_GAMES = 1000000

# Open the PGN file
pgn = open('lichess_db_standard_rated_2017-01.pgn')

# Read the first game
# first_game = chess.pgn.read_game(pgn)

games_1700_1900 = []
games_2400_2600 = []

count = 0
start_time = time.time()
while True:
    game = chess.pgn.read_game(pgn)
    if game is None or count >= NUM_GAMES:
        break
    if (1700 < int(game.headers["WhiteElo"]) < 1900) and (1700 < int(game.headers["BlackElo"]) < 1900):
        games_1700_1900.append(game)
    if (2400 < int(game.headers["WhiteElo"]) < 2600) and (2400 < int(game.headers["BlackElo"]) < 2600):
        games_2400_2600.append(game)
    count += 1

with open('chess_db_elo_1700_1900.pgn', 'w') as output_pgn:
    for game in games_1700_1900:
        exporter = chess.pgn.FileExporter(output_pgn)
        game.accept(exporter)
with open('chess_db_elo_2400_2600.pgn', 'w') as output_pgn:
    for game in games_2400_2600:
        exporter = chess.pgn.FileExporter(output_pgn)
        game.accept(exporter)

# Close the file when done
pgn.close()

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.4f} seconds")