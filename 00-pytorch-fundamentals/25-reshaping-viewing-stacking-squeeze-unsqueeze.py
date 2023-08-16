# https://www.learnpytorch.io/00_pytorch_fundamentals/#reshaping-stacking-squeezing-and-unsqueezing
# Reshaping, stacking, squeezing, and unsqueezing

# Reshaping - reshapes an input tensor to the shape of the output tensor
# View - return a view of an input tensor of certain shape but keep the same memory
# Stacking - concatenates a sequence of tensors along a new dimension (hstack - horizontal, vstack - vertical)
# Squeezing - removes dimensions of size 1 from the shape of a tensor
# Unsqueezing - adds a dimension of size 1 to the shape of a tensor
# Permute - Return a view of the input with the dimensions permuted (swapped) in a certain way

import torch

x = torch.arange(1., 13.) # torch.float32
print(x, x.shape) # torch.Size([10])

# Add an extra dimension (Must be compatible with the original shape)
x_reshaped = x.reshape(3, 4)
print(x_reshaped, x_reshaped.shape)

x_reshaped = x.reshape(1, 12)
print(x_reshaped, x_reshaped.shape)

x_reshaped = x.reshape(2, 6)
print(x_reshaped, x_reshaped.shape)

x_reshaped = x.reshape(12, 1)
print(x_reshaped, x_reshaped.shape)

x_reshaped = x.reshape(2, 2, 3)
print(x_reshaped, x_reshaped.shape)

try:
    x_reshaped = x.reshape(5,5) # RuntimeError: shape '[5, 5]' is invalid for input of size 12
except RuntimeError as e:
    print("Expected error: shape '[5, 5]' is invalid for input of size 12")

# Change the view of the tensor
z = x.view(3, 4)
print(z, z.shape)

# View is simmilar to reshape, but it keeps the same memory, so changing z will change x in this case
z[:, 0] = 5
print(z, x)

#Stacking

print("\n=== Stacking ===")
x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked, x_stacked.shape, "\n")

x_stacked = torch.stack([x, x, x, x], dim=1)
print(x_stacked, x_stacked.shape, "\n")


print("\n=== Squeezing / Unsqueezing ===")
x = torch.zeros(2, 1, 2, 1, 2)
print(x, x.shape)

x_squeezed = x.squeeze() # Removes all dimensions of size 1
print(x_squeezed, x_squeezed.shape)

# unsqueeze adds a dimension of size 1 at the specified dimension
x_squeezed_unsqueezed = x_squeezed.unsqueeze(dim=2) # Adds a dimension of size 1 at dim 2
print(x_squeezed_unsqueezed, x_squeezed_unsqueezed.shape) # torch.Size([2, 2, 1, 2])

# Permute rearranges the original tensor according to the specified dimensions
print("\n=== Permute ===")
image_tensor = torch.rand(size=(224,224,3)) # [height, width, color channels] - Much of deep learning is turning data into numbers
print(image_tensor.shape) # torch.Size([224, 224, 3])

# Permute the tensor to [color channels, height, width]
image_tensor_permuted = image_tensor.permute(2, 0, 1)
print(image_tensor_permuted.shape) # torch.Size([3, 224, 224])
