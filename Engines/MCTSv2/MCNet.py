import torch.nn as nn
import torch
import torch.nn.functional as F
import getBoard
import chess
import time

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class PolicyHead(nn.Module):
    def __init__(self, in_channels, board_size):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)  # 1x1 conv layer
        self.bn = nn.BatchNorm2d(2)
        self.fc_from = nn.Linear(2 * board_size * board_size, board_size * board_size)
        self.fc_to = nn.Linear(2 * board_size * board_size, board_size * board_size)

    def forward(self, x):
        pol_from = F.relu(self.bn(self.conv(x)))
        pol_from = pol_from.view(pol_from.size(0), -1)  # Flatten
        pol_from = self.fc_from(pol_from)

        pol_to = F.relu(self.bn(self.conv(x)))
        pol_to = pol_to.view(pol_to.size(0), -1)  # Flatten
        pol_to = self.fc_from(pol_to)
        return pol_from, pol_to

class ValueHead(nn.Module):
    def __init__(self, in_channels, board_size):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)  # 1x1 conv layer
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board_size * board_size, 256)
        self.fc2 = nn.Linear(256, 1)  # Scalar output

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))  # Scalar in range [-1, 1]

# Define a simple neural network for chess evaluation
class ChessEvaluator(nn.Module):
    def __init__(self, num_residual_blocks=39, input_channels=6, board_size=8):
        super(ChessEvaluator, self).__init__()
        self.conv = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)  # Input 6 channels (for the 6x8x8 board)
        self.bn = nn.BatchNorm2d(256)

        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(256, 256) for _ in range(num_residual_blocks)]
        )

        # Policy and Value heads
        self.policy_head = PolicyHead(256, board_size)
        self.value_head = ValueHead(256, board_size)

    def forward(self, x,x2):
        x = F.relu(self.bn(self.conv(x)))

        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Policy and Value outputs
        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value


if __name__ == '__main__':
    board = chess.Board(fen='rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4')
    test= ChessEvaluator()
    print(sum(p.numel() for p in test.parameters()))
    start_time= time.time()
    for i in range(1):
        evaluation = test(torch.Tensor(getBoard.get_bit_board(board)), torch.Tensor(getBoard.get_info_board(board)))
        print(evaluation)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")