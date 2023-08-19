import torch
from torch import nn

# Conv2D: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
torch.manual_seed(42)

# Create a batch of images
images = torch.randn(32, 3, 64, 64) # 32 images, 3 color channels, 64x64 pixels
test_image = images[0]

print(f"Test image: {test_image}\n")
print(f"Images shape: {images.shape}") # 32 images, 3 color channels, 64x64 pixels
print(f"\nOriginal Test image shape: {test_image.shape}") # 3 color channels, 64x64 pixels

conv_layer = nn.Conv2d(
    in_channels=3, # Color channels
    out_channels=10, # Hidden units
    kernel_size=3, # (3, 3) kernel / filter
    padding=1, # Add padding to the input image so the output image is the same size as the input
    stride=1)  # Move the kernel across the image by 1 pixel at a time

# Pass the test image through the convolutional layer
conv_output = conv_layer(test_image)
print(f"Convolutional output shape: {conv_output.shape}")

# nn.MaxPool2d: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
# Pooling layers are used to reduce the spatial dimensions of an image
# Max pooling takes the maximum value from a grid of pixels and uses that as the output
max_pool_layer = nn.MaxPool2d(kernel_size=2)
max_pool_output = max_pool_layer(conv_output)
print(f"MaxPool2d output shape: {max_pool_output.shape}\n")

torch.manual_seed(42)

image = torch.randn(1,1, 2, 2)
print(f"Image Tensor:\n{image}")
print(f"\nImages shape:\n{image.shape}")
print(f"\nAfter MaxPool2d:\n{nn.MaxPool2d(kernel_size=2)(image)}\n")
