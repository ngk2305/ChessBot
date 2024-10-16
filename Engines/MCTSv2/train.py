import torch
import chess
import torch.nn as nn
import torch.optim as optim
import chess_utils
import MCNet
import json
from torch.utils.data import Dataset, DataLoader
import getBoard
from tqdm import tqdm

class ChessDataset(Dataset):
    def __init__(self, games_data, evaluation_func):
        self.games_data = games_data
        self.evaluation_func = evaluation_func  # Function to evaluate positions

    def __len__(self):
        # Total number of plies across all games
        return sum(len(game['moves']) for game in self.games_data)

    def __getitem__(self, idx):
        # Find which game and ply the index corresponds to
        current_idx = 0
        for game in self.games_data:
            if current_idx + len(game['moves']) > idx:
                move_idx = idx - current_idx
                return self.get_ply_data(game, move_idx)
            current_idx += len(game['moves'])

    def get_ply_data(self, game, move_idx):
        board = chess.Board()

        # Play up to the current move
        for i, move in enumerate(game['moves'][:move_idx]):
            board.push_san(move)

        # Get board state
        board_matrix = torch.Tensor(getBoard.get_bit_board(board)),torch.Tensor(getBoard.get_info_board(board))

        # Get the move made at the current move
        move = game['moves'][move_idx]
        move = board.parse_san(move).uci()
        move = chess.Move.from_uci(move)

        move_from_to =move.from_square, move.to_square

        # Get evaluation (can be the game's result or an external evaluation function)
        evaluation = self.evaluation_func(board, game['result'])

        return board_matrix, torch.tensor(move_from_to), torch.tensor(evaluation)

def simple_evaluation_func(board, game_result):
    # Just return the game outcome for now
    if game_result == "1-0":
        return float(1)  # White wins
    elif game_result == "0-1":
        return float(0)  # Black wins
    else:
        return 0.5  # Draw


# Load data from the JSON file
def load_games_from_json(json_file):
    with open(json_file, 'r') as infile:
        games_data = json.load(infile)
    return games_data


class MCTS_Train:
    def __init__(self):
        self.net = MCNet.ChessEvaluator()
        self.episodes = 0
        self.score = 0



    def get_score(self,board):
        if board.outcome().winner:
            result = 1
        elif board.outcome().winner == None:
            result = 0.5
        else:
            result = 0
        return result


    def train_supervised(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")  # Use the first CUDA device
            print('cuda found')
        else:
            device = torch.device("cpu")  # Fallback to CPU if CUDA is not available

        self.net.to(torch.device("cuda:0"))

        criterion_move = nn.CrossEntropyLoss() # For move prediction (multi-class classification)
        criterion_eval = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        start_epoch = 50
        try:
            self.net.load_state_dict(torch.load(f'epoch{start_epoch-1}.pth'))
        except:
            print('Cant find any old weights')
        print('Training for 20')
        for i in range(start_epoch,start_epoch+15):
            print(f"Training Epoch number {i}")
            json_file = f"data/game_output{i}.json"
            games_data = load_games_from_json(json_file)
            chess_dataset = ChessDataset(games_data, simple_evaluation_func)
            batch_size = 32
            data_loader = DataLoader(chess_dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(1):
                for batch in tqdm(data_loader):
                    optimizer.zero_grad()
                    board_states, moves, evaluations= batch
                    evaluations = torch.tensor(evaluations,device=torch.device('cuda'))
                    main_board = torch.tensor(board_states[0],device=torch.device('cuda'))
                    xtra_info = torch.tensor(board_states[1], device=torch.device('cuda'))
                    
                    label_from = torch.tensor([move[0] for move in moves],device=torch.device('cuda'))
                    label_to = torch.tensor([move[1] for move in moves],device=torch.device('cuda'))


                    (model_from, model_to), eval = self.net(main_board,xtra_info)
                    print(criterion_move(model_from, label_from))
                    print(criterion_eval(evaluations,eval))
                    loss = criterion_move(model_from, label_from)+ criterion_move(model_to, label_to)+ criterion_eval(evaluations,eval)


                    loss.backward()
                    optimizer.step()
            torch.save(self.net.state_dict(), f'epoch{i}.pth')
            print(f'model saved to epoch{i}.pth')

    def train_self_play(self):
        running = True
        color = 0
        while running:
            if color == 0:
                white = self.current
                black = self.best
            else:
                white = self.best
                black = self.current

            board = chess.Board()
            while not board.is_game_over():
                if board.turn:
                    best_move, eval = white.find_best_move(board)
                    board.push(best_move)
                else:
                    best_move, eval = black.find_best_move(board)
                    board.push(best_move)

            result = self.get_score(board)
            self.current.train(result)

            color = 1 - color



if __name__ == '__main__':
    mcts = MCTS_Train()
    mcts.train_supervised()