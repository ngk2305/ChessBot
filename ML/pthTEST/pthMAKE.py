import torch

# Create example tensors
tensor1 = torch.randn(2, 3)
tensor2 = torch.randn(3, 4)

# Dictionary to save multiple tensors
tensors = {
    'tensor1': tensor1,
    'tensor2': tensor2
}

# Save the dictionary to a .pth file
torch.save(tensors, 'tensors.pth')

# Load the tensors from the .pth file
loaded_tensors = torch.load('tensors.pth')

# Access the loaded tensors
loaded_tensor1 = loaded_tensors['tensor1']
loaded_tensor2 = loaded_tensors['tensor2']

# Print the loaded tensors to verify
print("Loaded Tensor 1:", loaded_tensor1)
print("Loaded Tensor 2:", loaded_tensor2)