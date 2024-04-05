import pandas as pd
import torch
import BitboardExtraction
from NeuralNetworks import NeuralNetworkSuper
import numpy as np
import matplotlib.pyplot as plt

def main(datafile,model):

    data = pd.read_csv(f'Data/processedData3/{datafile}')
    X = data.drop('Score', axis=1).values


    predictions = []
    with torch.no_grad():
        for inputs in X:
            inputs = torch.tensor(BitboardExtraction.get_bit_fen_batch(inputs), dtype=torch.float32)
            output = model(inputs)
            predictions.append(output.item())

        # Add the predictions to the DataFrame
    data['predictions'] = predictions

    # Create subplots for each distribution
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(data['Score'], bins=20, color='blue', alpha=0.7)
    plt.title('Distribution of Score')

    plt.subplot(1, 2, 2)
    plt.hist(data['predictions'], bins=20, color='green', alpha=0.7)
    plt.title('Distribution of Pred')

    plt.tight_layout()
    plt.show()

    squared_diff = (data['Score'] - data['predictions']) ** 2

    # Calculate the mean squared error
    mse = squared_diff.mean()

    # Calculate the root mean squared error
    rmse = np.sqrt(mse)

    print("Root Mean Squared Error:", rmse)

        # Save the DataFrame with predictions to a new CSV file
    datafile = datafile.replace('.csv', "")
    data.to_csv(f'Data/Eval Data/{datafile}_1output_result.csv', index=False)

if __name__=='__main__':
    model = NeuralNetworkSuper.SuperChessEvaluator()
    model.load_state_dict((torch.load(f'Weights/super_model_weights.pth')))

    main('pData_20.csv',model)