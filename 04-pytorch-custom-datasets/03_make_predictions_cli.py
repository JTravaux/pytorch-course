import argparse
from torch import load
from models import FoodVisionMini
from torchvision import transforms
from matplotlib import pyplot as plt
from helper_functions import pred_and_plot_image

# Take in command line arguments
parser = argparse.ArgumentParser(description="Make predictions on a custom image")
parser.add_argument("-i", "--image", type=str, help="Path to the image to make a prediction on")
args = parser.parse_args()

image = args.image
print(f"Image path: {image}")
 
model = FoodVisionMini(input_shape=3, hidden_units=64, output_shape=3).to("cuda")
model.load_state_dict(load("models/04_food_vision_model_best.pth"))

pred_and_plot_image(model=model,
                    image_path=image,
                    class_names=["pizza", "steak", "sushi"],
                    transform=transforms.Compose([ transforms.Resize(size=(64, 64)) ]),
                    device="cuda")
plt.show()
