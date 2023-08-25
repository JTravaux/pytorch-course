import os
import numpy as np
from torch import load, nn
from models import FoodVisionMini, FoodVisionMiniV2
from torchvision import transforms, models
from matplotlib import pyplot as plt
from helper_functions import pred_image
from flask import Flask, redirect, url_for, request
from PIL import Image

app = Flask(__name__)
class_names = ["pizza", "steak", "sushi"]
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights).to("cuda") # FoodVisionMiniV2(input_shape=3, hidden_units=128, output_shape=3).to("cuda")

# WHen using pretrained models, we need to setup the classifier layer to match our needs
# We can freeze all of the layers/parameters in the features section by setting the attribute requires_grad=False.
for param in model.features.parameters():
    param.requires_grad = False

# Adjust output layer to our desired number of classes (3 in our case)
    # Note: Dropout layers randomly remove connections between two neural network layers with a probability of p. For example, if p=0.2, 20% of connections between neural network layers will be removed at random each pass. This practice is meant to help regularize (prevent overfitting) a model by making sure the connections that remain learn features to compensate for the removal of the other connections (hopefully these remaining features are more general).
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=len(class_names))
).to("cuda")

model.load_state_dict(load("models/FoodVisionMiniV2_EfficientNetB0.pth")) # 04_food_vision_model_v2_best

@app.route("/predict", methods=["POST"])
def predict():
    image = Image.open(request.files["image"])
    prediction = pred_image(model=model,
                        image=image,
                        class_names=class_names,
                        transform=weights.transforms(), #transforms.Compose([ transforms.Resize(size=(64, 64)), transforms.ToTensor() ]),
                        device="cuda")
    
    return {
        "prediction": prediction[0],
        "probability": prediction[1],
        "time_s": prediction[2],
        "time_ms": prediction[2] * 1000
    }

if __name__ == "__main__":
    app.run(debug=True)
