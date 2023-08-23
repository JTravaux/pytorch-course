import os
import random
import time
import zipfile
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from models import FoodVisionMini
from custom_dataset_class import ImageFolderCustom

import torch
import pandas as pd
from torch import nn
from torchinfo import summary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from helper_functions import walk_through_dir, train_model, eval_model, save_best_model, print_eval_results_table, plot_loss_curves
 
# Rough outline:
    # 1. Custom dataset (Pizza, Steak, and Sushi)
    # 2. Prepare and visualize the data
    # 3. Transforming data for use with CNN
    # 4. Loading custom data with prebuilt and custom functions
    # 5. Building "FoodVision Mini" model to classify pizza, steak, or sushi
    # 6. Comparing models with and without data augmentation
    # 7. Making predictions on custom images

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# 1. Custom dataset (Pizza, Steak, and Sushi)
# ============================================

# The dataset we will be using is a subset of the Food-101 dataset (https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
# "Start small and upgrade when necessary" to speed up experimentation time
# How dataset was created: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/04_custom_data_creation.ipynb

data_path = Path("data")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

BATCH_SIZE = 32

# Download the data if it doesn't exist
if image_path.is_dir() == False:
    print(f"{image_path} does not exist. Downloading data...")
    image_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "pizza_steak_sushi_20_percent.zip", "wb") as f:
        print("Downloading zip file...")
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")
        f.write(request.content)

    with zipfile.ZipFile(data_path / "pizza_steak_sushi_20_percent.zip", "r") as zip_ref:
        print("Unzipping file...")
        zip_ref.extractall(image_path)

    # Delete the zip file
    (data_path / "pizza_steak_sushi_20_percent.zip").unlink()

# ============================================
# 2. Prepare and visualize the data
# ============================================
walk_through_dir(image_path)

# Visualize an image (using PIL)
    # 1. Get all of the image paths
    # 2. Pick one at random (random.choice())
    # 3. Get the image class name using `pathlib.Path.parent.stem`
    # 4. Read the image using python's `PIL.Image.open()`
    # 5. Show image and print metadata
image_paths = list(image_path.glob("*/*/*.jpg")) # pizza_steak_sushi/*/*/*.jpg
random_image_path = random.choice(image_paths)
image_class = random_image_path.parent.stem
img = Image.open(random_image_path)
print(f"\nRandom image path: {random_image_path} | Image class: {image_class} | Image height: {img.height} | Image width: {img.width}\n")

# Visualize the image using matplotlib
# img_as_array = np.array(img)
# plt.figure(figsize=(10, 7))
# plt.imshow(img_as_array)
# plt.title(f"{image_class} - {img.height}x{img.width} - {img_as_array.shape}")
# plt.axis(False)
# plt.show()

# ============================================
# 3. Transforming data for use with CNN
# ============================================
# As we saw from above, the images are not all the same size, so we will need to resize them.
# Also, the color channel comes last, so we will need to transpose the images [H, W, C] -> [C, H, W] to match PyTorch's convention.
# Lastly, we will need to convert the images to tensors.
# train_transforms = transforms.Compose([ # nn.Sequential() also works
#     transforms.Resize(size=(64, 64)), # resize the images to 64x64
#     # transforms.RandomHorizontalFlip(p=0.5), # randomly flip the images horizontally (50% chance) (This is data augmentation)
#     transforms.ToTensor() # convert the images to PyTorch tensors, with color channel first [C, H, W]
# ])

# Let's take a look at one type used to train PyTorch vision models to state-of-the-art performance...
# more info: https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
# TrivialAugment: https://pytorch.org/vision/main/generated/torchvision.transforms.TrivialAugmentWide.html
train_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(), 
    transforms.ToTensor()
])

# Generally, for test data, we don't want to do any data augmentation
test_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# plot_transformed_images(image_paths=image_paths, transform=train_transforms, n=3, seed=None)
# ===================================================================
# 4 Loading custom data with prebuilt functions and custom functions
# ===================================================================

# Using the `torchvision.datasets.ImageFolder()` class
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=train_transforms, # Transform for the data
                                  target_transform=None) # Transform for the labels

test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms, target_transform=None)

class_names = train_data.classes
print(f"Number of training images: {len(train_data)}")
print(f"Number of test images: {len(test_data)}")
print(f"Class names: {class_names} | Class Indexes: {train_data.class_to_idx}\n")

img, label = train_data[0]
print(f"Image shape: {img.shape} | Image data type: {img.dtype} | Label: {label} | Class name: {class_names[label]} | Label data type: {type(label)}")

# Turn the data into batches (using `torch.utils.data.DataLoader()`)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# # ===================================================================
# # Example, this isn't needed in this case since we can use `torchvision.datasets.ImageFolder()`
# # However, this is useful if you have a dataset in a different format than the standard image classification format
# train_data_custom = ImageFolderCustom(dir_path=train_dir, transform=train_data_transform)
# test_data_custom = ImageFolderCustom(dir_path=test_dir, transform=test_data_transform)
# display_random_images(train_data_custom, train_data_custom.classes, n=3)

# # Can use the same `DataLoader` as above
# train_dataloader_custom = DataLoader(train_data_custom, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
# test_dataloader_custom = DataLoader(test_data_custom, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
# # ===================================================================

# Other forms of transforms (data augmentation)
# Data augmentation is a way of making your model more robust to unseen data by changing the training data slightly
# https://pytorch.org/vision/stable/transforms.html

# ============================================
# 5. Building "FoodVision Mini" model
# ============================================
EPOCHS = 50
eval_results_list = []

# Model 0: TinyVGG (baseline) with no data augmentation
model = FoodVisionMini(input_shape=3, hidden_units=128, output_shape=len(class_names)).to(device)
summary(model, input_size=(1, 3, 64, 64))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

res = train_model(
    model=model,
    epochs=EPOCHS,
    loss_fn=loss_fn,
    optimizer=optimizer,
    test_data=test_dataloader,
    train_data=train_dataloader,
    backups_during_training=False,
    device=next(model.parameters()).device,
)

eval_results_list.append(res["eval_results"])

plot_loss_curves(res["epoch_results"]) # https://www.learnpytorch.io/04_pytorch_custom_datasets/#8-what-should-an-ideal-loss-curve-look-like
print_eval_results_table(eval_results_list)
save_best_model(eval_results=eval_results_list, model_name="04_food_vision_model_best", models_dir="models", results_dir="data/pizza_steak_sushi")

# Some ways to evaluate the models:
    # 1. Hard coding
    # 2. Using tensorboard - https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
    # 3. Using weights and biases - https://wandb.ai/site/experiment-tracking
    # 4. MLFlow - https://mlflow.org/docs/latest/tracking.html

# model_df = pd.DataFrame(res["epoch_results"])

# # Plot the train loss
# plt.figure(figsize=(15, 10))
# plt.subplot(2, 2, 1)
# plt.plot(range(EPOCHS), model_df["train_loss"], label="Model 0")
# # ... rest of the models
# plt.title("Train Loss")
# plt.legend()

# # Plot the test loss
# plt.subplot(2, 2, 2)
# plt.plot(range(EPOCHS), model_df["test_loss"], label="Model 0")
# # ... rest of the models
# plt.title("Test Loss")
# plt.legend()

# # Plot the train accuracy
# plt.subplot(2, 2, 3)
# plt.plot(range(EPOCHS), model_df["train_acc"], label="Model 0")
# # ... rest of the models
# plt.title("Train Accuracy")
# plt.legend()

# # Plot the test accuracy
# plt.subplot(2, 2, 4)
# plt.plot(range(EPOCHS), model_df["test_acc"], label="Model 0")
# # ... rest of the models
# plt.title("Test Accuracy")
# plt.legend()

plt.show()
