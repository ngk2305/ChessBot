import pandas as pd
import torch
import BitboardExtraction
from NeuralNetworks import NeuralNetworkSuper

def main(datafile,model):

    data = pd.read_csv(f'Data/processedData/{datafile}')
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
    datafile = datafile.replace('.csv', "")
    data.to_csv(f'Data/Eval Data/{datafile}_result.csv', index=False)

if __name__=='__main__':
    model = NeuralNetworkSuper.SuperChessEvaluator()
    model.load_state_dict((torch.load(f'Weights/super_model_weights.pth')))

    main('pData_1.csv',model)