import torch.nn as nn
import torch
import torch.nn.functional as F


# Define a simple neural network for chess evaluation
class ChessEvaluator(nn.Module):
    def __init__(self):
        super(ChessEvaluator, self).__init__()

        # Convolutional layers with batch normalization and dropout
        self.conv1 = nn.Conv2d(6, 1024, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)


        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(1024 + 5, 512)  # Increased capacity
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x,x2):
        x = x.unsqueeze(0)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = x.view(-1, 256*4)  # flattening
        x2 = x2.view(-1,5)

        x = torch.cat((x,x2),dim=1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.sigmoid(self.fc4(x))
        return x

