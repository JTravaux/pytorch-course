import torch
import requests
import torchvision
from PIL import Image
from pathlib import Path
from models import FoodVisionMini
from matplotlib import pyplot as plt
from helper_functions import pred_and_plot_image

data_path = Path("data")
image_path = data_path / "pizza_steak_sushi"
classes = {'pizza': 0, 'steak': 1, 'sushi': 2}
idx_to_class = {v: k for k, v in classes.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = FoodVisionMini(input_shape=3, hidden_units=128, output_shape=3).to(device)
model.load_state_dict(torch.load("models/04_food_vision_model_best.pth"))

# Download (or use) custom image
custom_image_path = image_path / "custom_pizza.jpeg"
if custom_image_path.is_file() == False:
    print(f"{custom_image_path} does not exist. Downloading...")
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/blob/main/data/04-pizza-dad.jpeg?raw=true")
    with open(custom_image_path, "wb") as f:
        f.write(request.content)

# Make predictions on custom image
# We need to make sure the image is in the same format as the images the model was trained on
# In our case, this means:
    # * Tensor form with torch.float32 dtype
    # * Of shape 64x64x3
    # * On the correct device (GPU or CPU)

# custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32) / 255.0
# plt.imshow(custom_image_uint8.permute(1, 2, 0))
# plt.show()

# print(f"Custom image shape: {custom_image.shape}") # torch.Size([3, 4032, 3024])

# Resize the image
transform = torchvision.transforms.Compose([ torchvision.transforms.Resize(size=(64, 64)) ])
# custom_image_transformed = transform(custom_image)

# # Move the image to the correct device and add a batch dimension
# custom_image = custom_image_transformed.to(device).unsqueeze(0)

# Make a prediction
# model.eval()
# with torch.inference_mode():
#     custom_image_pred = model(custom_image)
#     custom_image_pred = torch.argmax(custom_image_pred, dim=1)
#     custom_image_label = idx_to_class[custom_image_pred.item()]
#     print(f"Custom image prediction: {custom_image_label}")

pred_and_plot_image(model=model, image_path=custom_image_path, class_names=["pizza", "steak", "sushi"], transform=transform, device=device)
plt.show()
