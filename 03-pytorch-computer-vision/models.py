import torch
from torch import nn

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(), # Without this, we'd get an error: RuntimeError: mat1 and mat2 shapes cannot be multiplied (28x28 and 784x32)
            nn.Linear(input_shape, hidden_units),
            nn.Linear(hidden_units, output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_shape),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer_stack(x)

# CNN (Convolutional Neural Network) Model
# CNN's are known to perform well on image data
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # 1 input channel, 32 output channels, 3x3 kernel size, 1 pixel padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2x2 kernel size, stride of 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # 32 input channels, 64 output channels, 3x3 kernel size, 1 pixel padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2x2 kernel size, stride of 2
            nn.Flatten(),
            nn.Linear(7*7*64, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_shape),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer_stack(x)
