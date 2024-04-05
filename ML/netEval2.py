import pandas as pd
import torch
import BitboardExtraction
from NeuralNetworks import NEv3
import torch.nn.functional as F
import numpy as np
def main(datafile,model):

    data = pd.read_csv(f'Data/processedData/{datafile}')
    X = data.drop('Score', axis=1).values


    predictions = []
    with torch.no_grad():
        for inputs in X:
            inputs = torch.tensor(BitboardExtraction.get_bit_fen_batch(inputs), dtype=torch.float32)
            output = model(inputs)
            probabilities = F.softmax(output, dim=1)

            score = probabilities[0][1]*0.5 + probabilities[0][2]


            predictions.append(score.item())

        # Add the predictions to the DataFrame
    data['predictions'] = predictions

    squared_diff = (data['Score'] - data['predictions']) ** 2

    # Calculate the mean squared error
    mse = squared_diff.mean()

    # Calculate the root mean squared error
    rmse = np.sqrt(mse)

    print("Root Mean Squared Error:", rmse)

        # Save the DataFrame with predictions to a new CSV file
    datafile = datafile.replace('.csv', "")
    data.to_csv(f'Data/Eval Data/{datafile}_result.csv', index=False)

if __name__=='__main__':
    model = NEv3.ChessEvaluator()
    model.load_state_dict((torch.load(f'Weights/NEv3.pth')))

    main('pData_20.csv',model)