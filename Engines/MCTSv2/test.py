import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use the first CUDA device
    print('cuda')
else:
    device = torch.device("cpu")  # Fallback to CPU if CUDA is not available

a=torch.cuda.FloatTensor()