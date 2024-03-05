import torch
from NeuralNetworks import NeuralNetworkSuper
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import BitboardExtraction

# Define your custom dataset class
class CustomDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # Add any necessary preprocessing steps here

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen = self.data.iloc[idx, 0]  # Assuming 'FEN' is the first column
        score = self.data.iloc[idx, 1]  # Assuming 'Score' is the second column

        # Add any necessary preprocessing steps here
        # For example, you might want to convert 'FEN' to a numerical representation

        sample = {'fen': fen, 'score': torch.tensor(score, dtype=torch.float)}
        return sample



# Instantiate your dataset and dataloader


# Instantiate your model, loss function, and optimizer
model = NeuralNetworkSuper.SuperChessEvaluator()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
num_epochs = 5
for j in range(1):
    print(j)
    data_path = f'Data/processedData/pData_{j+1}.csv'
    dataset = CustomDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1  , shuffle=True)


    for epoch in range(num_epochs):
        print(epoch)
        for batch in dataloader:
            fen = batch['fen']
            score = batch['score']
            fen = torch.tensor(BitboardExtraction.get_bit_fen_batch(fen), dtype=torch.float32)

            fen, score= fen.to(device), score.to(device)
            # Forward pass
            outputs = model(fen)
            loss = criterion(outputs, score)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model weights
    torch.save(model.state_dict(), f'Weights/super_model_weights.pth')
print('done')