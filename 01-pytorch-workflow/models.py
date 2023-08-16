import torch
from torch import nn

class LinearRegressionModel(nn.Module): # <--- nn.Module is the base class for all neural network modules in PyTorch
    """
    A PyTorch implementation of a linear regression model.
    """

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1)) # <--- requires_grad=True by default and float32 by default
        self.bias = nn.Parameter(torch.randn(1)) # <--- requires_grad=True by default and float32 by default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias # Same as linear regression formula (Y = a + bX) from above

class LinearRegressionModelV2(nn.Module): # <--- nn.Module is the base class for all neural network modules in PyTorch
    """
    A PyTorch implementation of a linear regression model.
    """

    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x) # Same as linear regression formula (Y = a + bX) from above
