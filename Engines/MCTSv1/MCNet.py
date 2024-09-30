import torch.nn as nn
import torch
import torch.nn.functional as F
import getBoard
import chess
import time

# Define a simple neural network for chess evaluation
class ChessEvaluator(nn.Module):
    def __init__(self):
        super(ChessEvaluator, self).__init__()
        self.conv1 = nn.Conv2d(6, 256, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(256 * 4 * 4 + 5, 512)  # assuming input size 8x8
        self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 1)

        self.piece_choose1 = nn.Linear(512, 256)
        self.piece_choose2 = nn.Linear(256, 64)
        self.piece_choose3 = nn.Linear(64, 6)

        self.move_choose1 = nn.Linear(512, 128)
        self.move_choose2 = nn.Linear(128, 32)
        self.move_choose3 = nn.Linear(32, 5)

    def forward(self, x,x2):
        x = self.pool(F.relu(self.conv1(x)))


        x = x.view(-1, 256 * 4 * 4)  # flattening
        x2 = x2.view(-1,5)
        x = torch.cat((x,x2),dim=1)

        x = F.relu(self.fc1(x))
        pc = F.relu(self.piece_choose1(x))
        pc = F.relu(self.piece_choose2(pc))
        pc = self.piece_choose3(pc)

        mc = F.relu(self.move_choose1(x))
        mc = F.relu(self.move_choose2(mc))
        mc = self.move_choose3(mc)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return pc, mc, x

if __name__ == '__main__':
    board = chess.Board(fen='rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4')
    test= ChessEvaluator()
    print(sum(p.numel() for p in test.parameters()))
    try:
        test.load_state_dict(torch.load('current.pth'))
    except:
        print('cant load')
    start_time= time.time()
    for i in range(1):
        evaluation = test(torch.Tensor(getBoard.get_bit_board(board)), torch.Tensor(getBoard.get_info_board(board)))
        print(evaluation)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")