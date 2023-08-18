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
            nn.Conv2d( # 2d is for two-dimensional data, such as images (our images have two dimensions: height and width. Yes, there's color channel dimension but each of the color channel dimensions have two dimensions too: height and width). For other dimensional data (such as 1D for text or 3D for 3D objects) there's also nn.Conv1d() and nn.Conv3d().
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3, # often also referred to as filter size, refers to the dimensions of the sliding window over the input. A smaller kernel size also leads to a smaller reduction in layer dimensions, which allows for a deeper architecture. Conversely, a large kernel size extracts less information, which leads to a faster reduction in layer dimensions, often leading to worse performance. Large kernels are better suited to extract features that are larger.
                padding=1, # Padding conserves data at the borders of activation maps, which leads to better performance, and it can help preserve the input's spatial size, which allows an architecture designer to build depper, higher performing networks
                stride=1), # Stride indicates how many pixels the kernel should be shifted over at a time. As stride is decreased, more features are learned because more data is extracted, which also leads to larger output layers. On the contrary, as stride is increased, this leads to more limited feature extraction and smaller output layer dimensions.
            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Takes the max value from a 2x2 grid of pixels.. purpose of gradually decreasing the spatial extent of the network, which reduces the parameters and overall computation of the network
        )

        # Feature extractor layer
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
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
