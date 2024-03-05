import torch.nn as nn

# Define a simple neural network for chess evaluation
class ChessEvaluator(nn.Module):
    def __init__(self):
        super(ChessEvaluator, self).__init__()
        self.fc1 = nn.Linear(64 * 12, 128)   # 64 squares for each of the 6 piece types
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, 64 * 12)  # Flatten the input bitboards
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

