import chess.pgn
import json


def parse_pgn(pgn_file):
    games_data = []

    # Open the PGN file
    with open(pgn_file, 'r') as pgn:
        # Loop through each game
        counter = 0
        count = 0
        while True:
            counter+=1
            print(counter)
            game = chess.pgn.read_game(pgn)
            if game is None:
                break  # End of file

            # Extract moves
            move_list = []
            board = game.board()
            for move in game.mainline_moves():

                move_list.append(board.san(move))
                board.push(move)

            # Extract result (1-0, 0-1, 1/2-1/2)
            result = game.headers["Result"]

            # Store game data
            game_data = {
                "moves": move_list,
                "result": result
            }

            if len(move_list) > 0:
                games_data.append(game_data)

            if counter%3000 == 0:
                output_file= f'game_output{count}.json'
                count+=1
                save_to_json(games_data, output_file)
                print(f'saved file number{count}')
                games_data = []





    return games_data


def save_to_json(games_data, output_file):
    with open(output_file, 'w') as outfile:
        json.dump(games_data, outfile, indent=4)


# Usage
pgn_file = "lichess_elite_2023-12.pgn"

games_data = parse_pgn(pgn_file)


print(f"Data saved")