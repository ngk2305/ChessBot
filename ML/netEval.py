import pandas as pd
import torch
import os
import importlib.util
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.A = data_dict['Bitboard']
        self.B = data_dict['Xtra']
        self.C = data_dict['Score']
        self.length = len(self.A)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Concatenate values from A and B
        A_value = self.A[idx]
        B_value = self.B[idx]
        board = [A_value, B_value]
        score = self.C[idx]
        return board, score

def main(datafile,model):

    data_dict = torch.load(datafile)
    dataset = CustomDataset(data_dict)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_eval = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input, score = batch
            board = torch.Tensor(input[0])
            xtra = torch.Tensor(input[1])
            score = score.float()

            board, score = board.to(device), score.to(device)
            # Forward pass
            outputs = model(board, xtra)

            true_eval.append(score.item())
            predictions.append(outputs.item())

            # Create a DataFrame from the two lists
    df = pd.DataFrame({
            'Eval': true_eval,
            'Pred': predictions
        })


    df.to_csv('eval.csv', index=False)
    average_abs_difference = np.mean(np.abs(np.subtract(true_eval, predictions)))
    print("Lists saved to eval.csv")
    print(f"The average of absolute differences is: {average_abs_difference}")

if __name__=='__main__':
    model_name = 'NEv5'
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'NeuralNetworks', model_name + '.py'))
    spec = importlib.util.spec_from_file_location(model_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ChessEvaluator()

    model.load_state_dict((torch.load(f'Weights/{model_name}_weights.pth')))

    main(f'Data/processedData4/pData_48.pth',model)