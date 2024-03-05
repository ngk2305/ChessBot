import torch.nn as nn

# Define a simple neural network for chess evaluation
class SuperChessEvaluator(nn.Module):
    def __init__(self):
        super(SuperChessEvaluator, self).__init__()
        self.fc1 = nn.Linear(64 * 12, 256)   # 64 squares for each of the 6 piece types
        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)


    def forward(self, x):
        x = x.view(-1, 64 * 12)  # Flatten the input bitboards
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

