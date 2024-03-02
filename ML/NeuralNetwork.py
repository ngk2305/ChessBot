import torch
import torch.nn as nn
import BitboardExtraction
import chess
import torch

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

if __name__=='__main__':
    # Create an instance of the neural network
    model = ChessEvaluator()
    model.load_state_dict(torch.load('model_weights.pth'))

    board = chess.Board()


    # Convert bitboards to PyTorch tensors
    bitboards = torch.tensor(BitboardExtraction.get_bit_board(board), dtype=torch.float32)

    # Forward pass through the neural network
    output_value = model(bitboards)

    # Print the output value
    print("Neural Network Output Value:", output_value.item())