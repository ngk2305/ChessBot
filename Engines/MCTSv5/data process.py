import chess.pgn
import json
import os

# Path to the directory containing your datasets
directory = 'data/'

# List all files in the directory
files = os.listdir(directory)

# Filter the files that match the pattern 'dataset+number'
dataset_files = [f for f in files if f.startswith('data_')]

# Extract the numbers from the filenames
numbers = [int(f.split('_')[1].replace('.json', '')) for f in dataset_files]

# Find the highest number
if numbers:
    current_max_number = max(numbers)
else:
    current_max_number = 0  # No datasets exist yet

# Increment to get the next number
next_number = current_max_number + 1

def parse_pgn(pgn_file):
    games_data = []

    # Open the PGN file
    with open(pgn_file, 'r') as pgn:
        # Loop through each game
        counter = 0
        count = next_number
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



            # Store game data
            game_data = {
                "moves": move_list,
            }

            if len(move_list) > 0:
                games_data.append(game_data)

            if counter%5000 == 0:
                output_file= f'data/data_{count}.json'
                count+=1
                save_to_json(games_data, output_file)
                print(f'saved file number{count}')
                games_data = []





    return games_data


def save_to_json(games_data, output_file):
    with open(output_file, 'w') as outfile:
        json.dump(games_data, outfile, indent=4)


# Usage
pgn_file = "lichess_elite_2022-05.pgn"

games_data = parse_pgn(pgn_file)


print(f"Data saved")