# https://www.learnpytorch.io/00_pytorch_fundamentals/#tensor-datatypes

import torch

# FLoat32 Tensor
float32_tensor = torch.tensor([3.0, 4.0],
                              dtype=None, # torch.float32 (default) 
                              device="cpu", # CPU (default), or "cuda" for GPU
                              requires_grad=False) # False (default) - Track the gradient with respect to this tensor's operations

print(float32_tensor.dtype) # torch.float32 (default)
print(float32_tensor.device) # CPU (default)
print(float32_tensor.requires_grad) # False (default)

float_16_tensor = float32_tensor.type(torch.float16) # or torch.half
print(float_16_tensor.dtype) # torch.float16
print(float_16_tensor.device) # cpu
print(float_16_tensor.requires_grad) # False (default)

float_64_tensor = float32_tensor.to(torch.float64) # or torch.double
# .to() is the same as .type(), the former is preferred

print(float_64_tensor.dtype) # torch.float64
print(float_64_tensor.device) # cpu
print(float_64_tensor.requires_grad) # False (default)
print(float_16_tensor * float_64_tensor) # tensor([ 9., 16.], dtype=torch.float64)


int_32_tensor = torch.tensor([3, 4], dtype=torch.int32)

# Some tensor attributes
print(int_32_tensor)
print(f"\nDevice of int_32_tensor: {int_32_tensor.device}")
print(f"Shape of int_32_tensor: {int_32_tensor.shape}") # or int_32_tensor.size()
print(f"Datatype of int_32_tensor: {int_32_tensor.dtype}")
