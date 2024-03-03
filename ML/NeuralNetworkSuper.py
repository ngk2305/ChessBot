import torch
import torch.nn as nn
import BitboardExtraction
import chess
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
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

if __name__=='__main__':
    # Create an instance of the neural network
    model = SuperChessEvaluator()
    model.load_state_dict((torch.load('super_model_weights.pth')))

    'rnbqkbnr/pppppppp/8/8/2BPPB2/2N2N2/PPP2PPP/R2QK2R w KQkq - 9 7'

    if 0:
        inputs = torch.tensor(BitboardExtraction.get_bit_fen('rnbqkbnr/pppppppp/8/8/2BPPB2/2N2N2/PPP2PPP/R2QK2R w KQkq - 9 7'), dtype=torch.float32)
        output = model(inputs)
        print(output)


    else:

        data = pd.read_csv(f'processedData/pData_3.csv')
        X = data.drop('Score', axis=1).values


        predictions = []
        with torch.no_grad():
            for inputs in X:
                inputs = torch.tensor(BitboardExtraction.get_bit_fen_batch(inputs), dtype=torch.float32)
                output = model(inputs)
                predictions.append(output.item())

        # Add the predictions to the DataFrame
        data['predictions'] = predictions

        # Save the DataFrame with predictions to a new CSV file
        data.to_csv('predictions_result3.csv', index=False)