import torch.nn as nn
import chess
import torch
import torch.nn.functional as F


# Define a simple neural network for chess evaluation
class ChessEvaluator(nn.Module):
    def __init__(self):
        super(ChessEvaluator, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 2 * 2 + 5, 256)  # assuming input size 8x8
        self.fc2 = nn.Linear(256, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x,x2):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 2 * 2)  # flattening
        x2 = x2.view(-1,5)
        x = torch.cat((x,x2),dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

