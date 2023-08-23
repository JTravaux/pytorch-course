import torch
from torch.nn import Module, Sequential, Conv2d, Linear, ReLU, MaxPool2d, Flatten

# Modeled after the TinyVGG architecture found here: https://poloclub.github.io/cnn-explainer/
class FoodVisionMini(Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = Sequential(
            Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1),
            ReLU(),
            Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = Sequential(
            Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1),
            ReLU(),
            Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )

        self.fully_connected_block = Sequential(
            Flatten(),
            Linear(in_features=hidden_units * 13 * 13, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fully_connected_block(self.conv_block_2(self.conv_block_1(x))) # Benefits from operator fusion

        