import torch.nn as nn
import torch
import torch.nn.functional as F


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

    def forward(self, x,x2):

        x = self.pool(F.relu(self.conv1(x)))


        x = x.view(-1, 256 * 4 * 4)  # flattening
        x2 = x2.view(-1,5)
        x = torch.cat((x,x2),dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x

