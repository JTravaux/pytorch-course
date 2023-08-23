import os
import numpy as np
from torch import load
from models import FoodVisionMini
from torchvision import transforms
from matplotlib import pyplot as plt
from helper_functions import pred_image
from flask import Flask, redirect, url_for, request
from PIL import Image

app = Flask(__name__)

model = FoodVisionMini(input_shape=3, hidden_units=64, output_shape=3).to("cuda")
model.load_state_dict(load("../models/04_food_vision_model_best.pth"))

@app.route("/predict", methods=["POST"])
def predict():
    image = Image.open(request.files["image"])
    prediction = pred_image(model=model,
                        image=image,
                        class_names=["pizza", "steak", "sushi"],
                        transform=transforms.Compose([ transforms.Resize(size=(64, 64)), transforms.ToTensor() ]),
                        device="cuda")
    
    return {
        "prediction": prediction[0],
        "probability": prediction[1],
        "time_taken": prediction[2],
    }

if __name__ == "__main__":
    app.run(debug=True)
