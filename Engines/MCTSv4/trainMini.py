import torch
import chess
import torch.nn as nn
import torch.optim as optim
import MCNetMini
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from chessLoader import ChessDataset

# Load data from the JSON file
def load_games_from_json(json_file):
    with open(json_file, 'r') as infile:
        games_data = json.load(infile)
    return games_data


class MCTS_Train:
    def __init__(self):
        self.net = MCNetMini.ChessEvaluator()
        self.episodes = 0
        self.score = 0


    def train_supervised(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")  # Use the first CUDA device
            print('cuda found')
        else:
            device = torch.device("cpu")  # Fallback to CPU if CUDA is not available

        self.net.to(torch.device("cuda:0"))

        criterion_move = nn.CrossEntropyLoss()  # For move prediction (multi-class classification)
        optimizer = optim.Adam(self.net.parameters(), lr=0.0005)
        start_epoch = 25
        try:
            self.net.load_state_dict(torch.load(f'epoch{start_epoch - 1}.pth'))
        except:
            print('Cant find any old weights')
        print('Training for 20')
        for i in range(start_epoch, start_epoch + 25):
            print(f"Training Epoch number {i}")
            json_file = f"data/data_{i}.json"
            games_data = load_games_from_json(json_file)
            chess_dataset = ChessDataset(games_data)
            batch_size = 64
            data_loader = DataLoader(chess_dataset, batch_size=batch_size, shuffle=True)
            running_train_loss = 0.0
            for batch in tqdm(data_loader):
                optimizer.zero_grad()
                board_states, moves= batch

                main_board = torch.tensor(board_states[0], device=torch.device('cuda'))
                xtra_info = torch.tensor(board_states[1], device=torch.device('cuda'))

                label_from = torch.tensor([move[0] for move in moves], device=torch.device('cuda'))
                label_to = torch.tensor([move[1] for move in moves], device=torch.device('cuda'))

                model_from, model_to = self.net(main_board, xtra_info)


                loss = criterion_move(model_from, label_from) + criterion_move(model_to, label_to)
                running_train_loss += loss.item() * len(board_states[0])

                loss.backward()
                optimizer.step()

            avg_train_loss = running_train_loss / (len(data_loader) * batch_size)
            print(f"Epoch {i}, Training Loss: {avg_train_loss:.4f}")
            torch.save(self.net.state_dict(), f'epoch{i}.pth')
            print(f'model saved to epoch{i}.pth')



if __name__ == '__main__':
    mcts = MCTS_Train()
    mcts.train_supervised()