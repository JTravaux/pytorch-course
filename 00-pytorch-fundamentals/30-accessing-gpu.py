# https://www.learnpytorch.io/00_pytorch_fundamentals/#running-tensors-on-gpus-and-making-faster-computations
# Running Tensors on GPUs and Making Faster Computations

import torch

# Set device to GPU & setup device agnostic code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Note: cuda:0 is the first GPU, helpful if you have multiple GPUs
print(device)

# Count number of GPUs
print(torch.cuda.device_count())
