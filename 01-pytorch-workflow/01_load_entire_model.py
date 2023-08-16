import torch
from models import LinearRegressionModel # Need to import the LinearRegressionModel class from LinearRegressionModel.py

# Load the entire model
model = torch.load('models/01_first_pytorch_workflow_model_0.pth')
model.eval()

print(model.state_dict())
