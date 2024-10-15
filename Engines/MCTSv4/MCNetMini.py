import torch.nn as nn
import torch
import torch.nn.functional as F
import getBoard
import chess
import time

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_prob=0.5):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Apply dropout after the first conv layer
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class PolicyHead(nn.Module):
    def __init__(self, in_channels, board_size, dropout_prob=0.5):
        super(PolicyHead, self).__init__()
        self.out_channel = 32
        self.conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1)  # 1x1 conv layer
        self.conv2 = nn.Conv2d(in_channels, self.out_channel, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.out_channel)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc_from = nn.Linear(self.out_channel * board_size * board_size+5, board_size * board_size)
        self.fc_to = nn.Linear(self.out_channel * board_size * board_size+5, board_size * board_size)

    def forward(self, x, x2):
        pol_from = F.relu(self.bn(self.conv(x)))
        pol_from = self.dropout(pol_from)  # Apply dropout
        pol_from = pol_from.view(pol_from.size(0), -1)  # Flatten
        pol_from = torch.cat((pol_from, x2), dim=1)
        pol_from = self.fc_from(pol_from)

        pol_to = F.relu(self.bn(self.conv2(x)))
        pol_to = self.dropout(pol_to)  # Apply dropout
        pol_to = pol_to.view(pol_to.size(0), -1)  # Flatten
        pol_to = torch.cat((pol_to, x2), dim=1)
        pol_to = self.fc_from(pol_to)
        return pol_from, pol_to

# Define a simple neural network for chess evaluation
class ChessEvaluator(nn.Module):
    def __init__(self, num_residual_blocks=5, input_channels=6, board_size=8, dropout_prob=0.5):
        super(ChessEvaluator, self).__init__()
        self.channel = 256
        self.conv = nn.Conv2d(input_channels, self.channel, kernel_size=3, padding=1)  # Input 6 channels (for the 6x8x8 board)
        self.bn = nn.BatchNorm2d(self.channel)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer after the initial conv layer

        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(self.channel, self.channel) for _ in range(num_residual_blocks)]
        )

        # Policy and Value heads
        self.policy_head = PolicyHead(self.channel, board_size)


    def forward(self, x,x2):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            x2 = x2.unsqueeze(0)
        x = F.relu(self.bn(self.conv(x)))
        x = self.dropout(x)
        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Policy and Value outputs
        policy = self.policy_head(x,x2)


        return policy


if __name__ == '__main__':
    board = chess.Board(fen='r3k2r/1ppbqpp1/p1n1p2p/8/3PN1n1/2PB1NP1/PP2QPP1/2KR3R w kq - 1 14')
    test= ChessEvaluator()
    try:
        test.load_state_dict(torch.load('epoch8.pth'))
    except:
        print('cant load')
    print(sum(p.numel() for p in test.parameters()))
    start_time= time.time()
    for i in range(1):
        evaluation = test(torch.Tensor(getBoard.get_bit_board(board)), torch.Tensor(getBoard.get_info_board(board)))
        print(evaluation)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")