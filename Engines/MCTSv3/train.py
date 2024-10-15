import torch
import torch.nn as nn
import torch.optim as optim
import MCNet
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from chessLoader import ChessDataset
from torch.cuda.amp import autocast, GradScaler

def load_games_from_json(json_file):
    with open(json_file, 'r') as infile:
        games_data = json.load(infile)
    return games_data


class MCTS_Train:
    def __init__(self):
        self.net = MCNet.ChessEvaluator()
        self.episodes = 0
        self.score = 0


    def train_supervised(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")  # Use the first CUDA device
            print('cuda found')
        else:
            device = torch.device("cpu")  # Fallback to CPU if CUDA is not available

        self.net.to(torch.device("cuda"))

        criterion_move = nn.CrossEntropyLoss() # For move prediction (multi-class classification)
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        start_epoch = 0
        try:
            self.net.load_state_dict(torch.load(f'epoch{start_epoch-1}.pth'))
        except:
            print('Cant find any old weights')
        print('Training for 20')

        scaler = GradScaler()
        avg_train_loss = 0.0

        for i in range(start_epoch,start_epoch+20):
            print(f"Training Epoch number {i}")
            json_file = f"data/data_{i}.json"
            games_data = load_games_from_json(json_file)
            chess_dataset = ChessDataset(games_data)
            batch_size = 512
            data_loader = DataLoader(chess_dataset, batch_size=batch_size, shuffle=True)

            if not (i%5 == 4):
                self.net.train()
                running_train_loss = 0.0
                if (i%5 == 0):
                    avg_train_loss = 0.0
                for batch in tqdm(data_loader):
                    optimizer.zero_grad()
                    board_states, moves = batch

                    main_board = torch.tensor(board_states[0],device=torch.device('cuda'))
                    xtra_info = torch.tensor(board_states[1], device=torch.device('cuda'))
                    
                    label_from = torch.tensor([move[0] for move in moves],device=torch.device('cuda'))
                    label_to = torch.tensor([move[1] for move in moves],device=torch.device('cuda'))

                    # Forward pass with autocast (mixed precision)
                    with autocast():
                        model_from, model_to = self.net(main_board,xtra_info)
                        loss = criterion_move(model_from, label_from)+ criterion_move(model_to, label_to)

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()

                    # Gradient update step
                    scaler.step(optimizer)
                    scaler.update()

                    running_train_loss += loss.item() * len(board_states[0])
                avg_train_loss += running_train_loss / (len(data_loader)*512) /4
            else:
                self.net.eval()
                running_val_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(data_loader):
                        optimizer.zero_grad()
                        board_states, moves = batch

                        main_board = torch.tensor(board_states[0], device=torch.device('cuda'))
                        xtra_info = torch.tensor(board_states[1], device=torch.device('cuda'))

                        label_from = torch.tensor([move[0] for move in moves], device=torch.device('cuda'))
                        label_to = torch.tensor([move[1] for move in moves], device=torch.device('cuda'))

                        # Forward pass with autocast (mixed precision)
                        with autocast():
                            model_from, model_to = self.net(main_board, xtra_info)
                            loss = criterion_move(model_from, label_from) + criterion_move(model_to, label_to)

                        running_val_loss += loss.item() * len(board_states[0])

                avg_val_loss = running_val_loss / (len(data_loader)* 512)

                print(f"Epoch {(i+1-5)/5}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
                torch.save(self.net.state_dict(), f'epoch{(i+1-5)/5}.pth')
                print(f'model saved to epoch{(i+1-5)/5}.pth')




if __name__ == '__main__':
    mcts = MCTS_Train()
    mcts.train_supervised()