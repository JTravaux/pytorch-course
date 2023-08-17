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
# Typical structure of a convolutional neural network: Input layer -> [Convolutional layer -> activation layer -> pooling layer] -> Output layer
# Where the contents of [Convolutional layer -> activation layer -> pooling layer] can be upscaled and repeated multiple times, depending on requirements (more layers = deeper network)
class FashionMNISTModelV2(nn.Module):
    """Model architecture that replicates the TinyVGG model from the CNN explainer website: https://poloclub.github.io/cnn-explainer/"""
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        # Feature extractor layer
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Takes the max value from a 2x2 grid of pixels
        )

        # Feature extractor layer
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Classifier layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, # 7x7 is the output shape of the second convolutional layer
                      out_features=output_shape),
        )


    def forward(self, x):
        x = self.conv_block_1(x) # Shape after conv_block_1: torch.Size([32, hidden_units, 14, 14])
        x = self.conv_block_2(x) # Shape after conv_block_1: torch.Size([32, hidden_units, 7, 7])
        x = self.classifier(x) # Shape after classifier: torch.Size([32, output_shape])
        return x
