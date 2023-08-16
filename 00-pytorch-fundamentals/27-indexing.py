# https://www.learnpytorch.io/00_pytorch_fundamentals/#indexing-selecting-data-from-tensors
# Indexing / Selecting data from tensors

import torch

x = torch.arange(1,10).reshape(1,3,3)
print(x, x.shape)

# Index on tensor on 0th dimension
print(x[0], x[0].shape)

# Index on tensor on 1st dimension
print(x[0, 0], x[0, 0].shape) # or print(x[0][0], x[0][0].shape)

# Index on tensor on 2nd dimension
print(x[0, 0, 0], x[0, 0, 0].shape, "\n") # or print(x[0][0][0], x[0][0][0].shape)


# You can also use ":" to select all elements in a dimension
print(x[:, 0]) # Select all elements in the 0th dimension = tensor([[1, 2, 3]])

# Get all values of 0th and 1st dimension, but only index 1 of the 2nd dimension
print(x[:, :, 1]) # tensor([[2, 5, 8]])
print(x[:, :, 2]) # tensor([[3, 6, 9]])

# Get alll values of the 0 dimension, vbut only the 1 index value of the first and second dimension
print(x[:, 1, 1]) # tensor([5])

# Get index 0 of the 0th and 1st dimension and all values of the 2nd dimension
print(x[0, 0, :]) # tensor([1, 2, 3]), which is the same as print(x[0, 0])
