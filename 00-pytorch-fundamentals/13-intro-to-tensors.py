# https://www.learnpytorch.io/00_pytorch_fundamentals/#creating-tensors

import torch

# Scalar
print("SCALAR")
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim) # 0
print(scalar.item()) # 7

# Vector
print("\nVECTOR")
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim) # 1
print(vector.shape) # torch.Size([2])
# print(vector.item()) # Error: only one element tensors can be converted to Python scalars

# MATRIX
print("\nMATRIX")
MATRIX = torch.tensor([
    [7, 8],
    [9, 10]
])

print(MATRIX)
print(MATRIX.ndim) # 2
print(MATRIX.shape) # torch.Size([2, 2])
print(MATRIX[0]) # tensor([7, 8])
print(MATRIX[0][0]) # tensor(7)

# TENSOR
print("\nTENSOR")
TENSOR = torch.tensor([
    [ # Matrix 1
        [ # Row 1
            1, # Column 1
            2, # Column 2
            3, # Column 3
        ],
        [4, 5, 6]
    ],
    [ # Matrix 2
        [7, 8, 9],
        [10, 11, 12]
    ],
    [ # Matrix 3
        [13, 14, 15],
        [16, 17, 18]
    ]
])

print(TENSOR)
print(TENSOR.ndim) # 3

print(TENSOR.shape) # torch.Size([3, 2, 3])
# In the above, the first number is the number of matrices (3), the second is the number of rows in each (2), and the third is the number of columns in each row (3)

print(TENSOR[0]) # tensor([[1, 2, 3], [4, 5, 6]])
print(TENSOR[0][0]) # tensor([1, 2, 3])
print(TENSOR[0][0][0]) # tensor(1)

# Random Tensors
print("\n======RANDOM TENSORS======")
# Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers...
# This is the process of training a neural network

RANDOM1 = torch.rand(3, 4)
print(RANDOM1)
print(RANDOM1.shape) # torch.Size([3, 4])
print(RANDOM1.ndim) # 2

# RANDOM2 = torch.rand(2, 2, 10)
# print(RANDOM2)
# print(RANDOM2.shape) # torch.Size([2, 2, 10])
# print(RANDOM2.ndim) # 3

# RANDOM3 = torch.rand(2, 2, 10, 10)
# print(RANDOM3)
# print(RANDOM3.shape) # torch.Size([2, 2, 10, 10])
# print(RANDOM3.ndim) # 4


random_image_size_tensor = torch.rand(size=(224, 224, 3)) # 224x224 image with 3 color channels (RGB)
print(random_image_size_tensor.shape) # torch.Size([224, 224, 3])
print(random_image_size_tensor.ndim) # 3

# Zero and Ones Tensors
print("\n======ZERO AND ONES TENSORS======")
zero = torch.zeros(3, 4)
print(zero)
print(zero.shape) # torch.Size([3, 4])
print(zero.ndim) # 2
print(zero * RANDOM1)

ones = torch.ones(3, 4)
print(ones)
print(ones.shape) # torch.Size([3, 4])
print(ones.ndim) # 2
print(ones * RANDOM1)
print(ones.dtype) # torch.float32

# Tensors in a range
print("\n======TENSORS IN A RANGE======")
range_tensor = torch.arange(start=0, end=10, step=2)
print(range_tensor) # tensor([0, 2, 4, 6, 8])
print(range_tensor.shape) # torch.Size([5])
print(range_tensor.ndim) # 1
print(range_tensor.dtype) # torch.int64

range_tensor_like = torch.zeros_like(input=range_tensor)
print(range_tensor_like) # tensor([0, 0, 0, 0, 0])
print(range_tensor_like.shape) # torch.Size([5])
print(range_tensor_like.ndim) # 1
print(range_tensor_like.dtype) # torch.int64
