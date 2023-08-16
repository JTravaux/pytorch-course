# https://www.learnpytorch.io/00_pytorch_fundamentals/#matrix-multiplication-is-all-you-need
# http://matrixmultiplication.xyz/

import torch

# Two main ways of performing multiplication in neural networks:
# 1. Element-wise multiplication: Multiply each element of one tensor with the corresponding element of another tensor
# 2. Matrix multiplication / Dot product (most common in NN's): Multiply a matrix with another matrix, or a matrix with a vector, or a matrix with a scalar

# Element-wise multiplication
print("=== Element-wise multiplication ===")
tensor1 = torch.tensor([1, 2, 3])
print(tensor1, "*", tensor1, "=", tensor1 * tensor1) # tensor([1, 4, 9])

print("\n=== Matrix multiplication ===")
print(tensor1, "•", tensor1, "=", tensor1.matmul(tensor1)) # tensor(14) / which means 1*1 + 2*2 + 3*3 = 14
# the @ operator is also used for matrix multiplication, however matmul is more explicit

print("\n=== Rules of matrix multiplication ===")
# Rules of matrix multiplication
# One of the most common errors in matrix multiplication is the shape of the tensors

# 1. The inner dimensions of the matrices must match
# (3, 2) @ (3, 2) = Error
# (3, 2) @ (2, 3) = (3, 3)
rand1 = torch.rand(3, 2) # torch.Size([3, 2])
rand2 = torch.rand(2, 3) # torch.Size([2, 3])

try:
    print(rand1.matmul(rand1)) # RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)
except RuntimeError as e:
    print("Expected: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)")

# 2. The resulting matrix has the shape of the outer dimensions
# (3, 2) @ (2, 3) = (3, 3) - 2 is the inner dimension, 3 is the outer dimension
print(rand1.matmul(rand2)) # torch.Size([3, 3])

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]]) # torch.Size([3, 2])

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]]) # torch.Size([3, 2])

try:
    torch.matmul(tensor_A, tensor_B) # torch.Size([3, 2]) or torch.mm() for short - resullts in an error
except RuntimeError as e:
    print("Expected error")

print("\n=== Manipulate Tensor Shapes ===")
# A transpose of a matrix switches the axes or dimensions of a matrix
print(tensor_B.T.shape) # torch.Size([2, 3]) now
print(tensor_A.matmul(tensor_B.T)) # torch.Size([3, 3])
