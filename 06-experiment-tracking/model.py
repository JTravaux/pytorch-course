import torch
from torch import nn
from torchvision import models

# Get the FoodVisionMini current best model
def get_model(output_size = 3, device = "cuda"):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights).to("cuda") 

    for param in model.features.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=output_size)
    ).to(device)

    model.load_state_dict(torch.load("models/FoodVisionMiniV2_EfficientNetB0.pth"))

    return model, weights.transforms()
