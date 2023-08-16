# https://www.learnpytorch.io/00_pytorch_fundamentals/#manipulating-tensors-tensor-operations

import torch

int_32_tensor = torch.tensor([3, 4], dtype=torch.int32)
print(f"Device of int_32_tensor: {int_32_tensor.device}")
print(f"Shape of int_32_tensor: {int_32_tensor.shape}") # or int_32_tensor.size()
print(f"Datatype of int_32_tensor: {int_32_tensor.dtype}")

# Change the device of a tensor
print("\nChange the device of a tensor")
print(int_32_tensor.to("cuda")) # tensor([3, 4], device='cuda:0', dtype=torch.int32)

# Change the datatype of a tensor
print("\nChange the datatype of a tensor")
print(int_32_tensor.to(torch.float64)) # tensor([3., 4.], dtype=torch.float64)

# Change the datatype and device of a tensor
print("\nChange the datatype and device of a tensor")
print(int_32_tensor.to(torch.int32).to("cpu")) # tensor([3., 4.], device='cuda:0', dtype=torch.float64)

# Tensor operations
print("\n=== Tensor operations ===")

tensor = torch.tensor([1,2,3])
print(tensor) # tensor([1, 2, 3])

print(tensor + 10) # tensor([11, 12, 13])
# print(tensor.add(10)) # tensor([11, 12, 13]
# print(torch.add(tensor, 10)) # tensor([11, 12, 13]

print(tensor - 10) # tensor([-9, -8, -7])
# print(tensor.sub(10)) # tensor([-9, -8, -7])
# print(torch.sub(tensor, 10)) # tensor([-9, -8, -7])

print(tensor * 10) # tensor([10, 20, 30])
# print(tensor.mul(10)) # tensor([10, 20, 30])
# print(torch.mul(tensor, 10)) # tensor([10, 20, 30])

print(tensor / 10) # tensor([0.1000, 0.2000, 0.3000])
# print(tensor.div(10)) # tensor([0.1000, 0.2000, 0.3000])
# print(torch.div(tensor, 10)) # tensor([0.1000, 0.2000, 0.3000])
