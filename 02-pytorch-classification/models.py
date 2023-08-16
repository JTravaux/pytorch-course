from torch import nn

# 2.1. Subclassing the nn.Module class
# 2.2. Creater two nn.Linear layers
# 2.3. Define the forward method
class CircleModelV1(nn.Module):
    """
    A binary classification model with 2 inputs and 1 output.
    """

    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=2, out_features=8) # Gernerally, the more hidden layers the better the model will be at learning
        self.linear_2 = nn.Linear(in_features=8, out_features=1) 

    def forward(self, X):
        return self.linear_2(self.linear_1(X)) # x -> linear_1 -> linear_2 -> output
    
class CircleModelV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10) # Added more hidden units
        self.layer_2 = nn.Linear(in_features=10, out_features=10) # Added another layer
        self.layer_3 = nn.Linear(in_features=10, out_features=1) 

    def forward(self, x):
        # z = self.layer_1(x) # z = logits
        # z = self.layer_2(z)
        # z self.layer_3(z)
        # return z
        return self.layer_3(self.layer_2(self.layer_1(x))) # x -> layer_1 -> layer_2 -> layer_3 -> output (leverages speedups where possible)
     
class CircleModelV2Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=1, out_features=10) # Added more hidden units
        self.layer_2 = nn.Linear(in_features=10, out_features=10) # Added another layer
        self.layer_3 = nn.Linear(in_features=10, out_features=1) 

    def forward(self, x):
        # z = self.layer_1(x) # z = logits
        # z = self.layer_2(z)
        # z self.layer_3(z)
        # return z
        return self.layer_3(self.layer_2(self.layer_1(x))) # x -> layer_1 -> layer_2 -> layer_3 -> output (leverages speedups where possible)
     
class CircleModelV3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # Added a non-linear activation function
        # We could also use a sigmoid function here, rather than doing the sigmoid in the loss function

    def forward(self, x):
        # Where should we put out non-linear activation function?
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))) # x -> layer_1 -> relu -> layer_2 -> relu -> layer_3 -> output (leverages speedups where possible) 

# Multiclass classification model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8) -> None:
        """A multiclass classification model with input_feature inputs and output_features outputs.

        Args:
            input_feature (int): Number of input features
            output_features (int): Number of output features
            hidden_units (int, optional): Number of hidden units. Defaults to 8.
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            # nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            # nn.ReLU(),
            nn.Linear(hidden_units, output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
