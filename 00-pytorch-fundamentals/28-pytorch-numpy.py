# https://www.learnpytorch.io/00_pytorch_fundamentals/#pytorch-tensors-numpy
# PyTorch Tensors & Numpy

import torch
import numpy as np

# default numpy array type is float64
# default PyTorch tensor type is float32

# NumPy array to PyTorch tensor
array = np.array([1.0, 2.0, 3.0])
print(array, type(array))

tensor = torch.from_numpy(array)
print(tensor, type(tensor))

# PyTorch tensor to NumPy array
numpy_tensor = tensor.numpy()
print(numpy_tensor, type(numpy_tensor))

# Doesn't share memory (either direction)
tensor = tensor + 1
print(tensor, numpy_tensor)
