import torch
from NeuralNetworks import NEv3
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import BitboardExtraction
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
start_time = time.time()


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NEv3.ChessEvaluator()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.to(device)

for j in range(1):
    data_path = f'Data/processedData2/pData_{j+5}.csv'
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Score']).values
    y = data['Score'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = [BitboardExtraction.get_bit_fen_batch(x) for x in X_train]
    X_test = [BitboardExtraction.get_bit_fen_batch(x)  for x in X_test]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)



    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Training loop
    num_epochs = 1

    for epoch in range(num_epochs):
        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            #To cuda
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            try:
                loss = criterion(outputs, labels)
            except:
                print(inputs)
                print(outputs)
                print(labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # Evaluate the model
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f'Accuracy: {accuracy}')

    # Save the trained model weights
    torch.save(model.state_dict(), f'Weights/NEv3.pth')

print('done')
print("--- %s seconds ---" % (time.time() - start_time))
