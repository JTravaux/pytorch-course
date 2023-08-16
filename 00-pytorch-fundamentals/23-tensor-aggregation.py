# https://www.learnpytorch.io/00_pytorch_fundamentals/#finding-the-min-max-mean-sum-etc-aggregation
# Finding the min, max, mean, sum, etc. (aggregation)

import torch

x = torch.arange(1, 100, 10)
print(x)
print(x.dtype) # torch.int64

print(f"Min: {x.min()}")
print(f"Max: {x.max()}")
print(f"Sum: {x.sum()}")

try:
    print(f"Mean: {x.mean()}")
except RuntimeError as e:
    print("Expected error: Input dtype should be either floating point or complex dtypes. Got Long instead.")
    
print(f"Mean: {x.type(torch.float32).mean()}") # torch.float32 or print(f"Mean: {x.float().mean()}")

print("\n=== Positional Min/Max ===")
print(x.argmin()) # tensor(0)
print(x.argmax()) # tensor(9)
