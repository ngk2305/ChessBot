import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import importlib.util
import os

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

# Import Neural Network

module_name = 'NEv5'
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'NeuralNetworks', module_name+'.py'))
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Instantiate your model, loss function, and optimizer
model = module.ChessEvaluator()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
num_epochs = 1
dataFolder = 4
model.train()
for j in range(1,10):
    data_path = f'Data/processedData{dataFolder}/pData_{j}.pth'
    data_dict = torch.load(data_path)
    dataset = CustomDataset(data_dict)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            optimizer.zero_grad()

            input, score = batch

            board= torch.Tensor(input[0])
            xtra = torch.Tensor(input[1])
            score= score.float()

            board, score= board.to(device), score.to(device)
            # Forward pass
            outputs = model(board,xtra)

            loss = criterion(outputs, score)

            # Backward pass and optimization

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')



    torch.save(model.state_dict(), f'Weights/{module_name}_weights.pth')
    print('model saved')
print('done')
