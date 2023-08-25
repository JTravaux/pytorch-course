# Transfer learning allows us to take the patterns (also called weights) another model has learned from another problem and use them for our own problem.
# This often means we can get away with less data and/or less compute to achieve similar results to models built from scratch.

# For this, we will be using a model trained on ImageNet for our FoodVision Mini model.
# Or we could take the patterns from a language model (a model that's been through large amounts of text to learn a representation of language) and use them as the basis of a model to classify different text samples.

# The premise remains: find a well-performing existing model and apply it to your own problem.

# Places to get pre-trained models:
    # 1. PyTorch models (https://pytorch.org/vision/stable/models.html)
    # 2. HuggingFace Hub (https://huggingface.co/models)
    # 3. timm (Pytorch Image Models) library (https://github.com/huggingface/pytorch-image-models)
    # 4. Papers with code (https://paperswithcode.com/)

import os
import torch
import torchvision
from torch import nn
from pathlib import Path
from torchinfo import summary
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from helper_functions import download_data, create_dataloaders, train_model, print_eval_results_table, plot_loss_curves, save_best_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# 1. Get Data & Transforms Ready
# ============================================
BATCH_SIZE = 32
MODEL_NAME = "FoodVisionMiniV2_EfficientNetB0"

folder = "pizza_steak_sushi"
image_path = Path("data/") / folder
train_dir = image_path / "train"
test_dir = image_path / "test"

download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip", destination=folder) # 10%: https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip

# When using a pretrained model, it's important that your custom data going into the model is prepared in the same way as the original training data that went into the model.
# Since we'll be using a pretrained model from torchvision.models, there's a specific transform we need to prepare our images first...
    # With pytorch 0.13+, there's a new way to do this called "automatic creation". We'll try both.

# 1. Manual Creation
    # Prior to torchvision v0.13+, to create a transform for a pretrained model in torchvision.models, the documentation stated something like:
        # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
        # The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

        # Where did the mean and standard deviation values come from? Why do we need to do this?
            # These were calculated from the data. Specifically, the ImageNet dataset by taking the means and standard deviations across a subset of images.
            # We also don't need to do this. Neural networks are usually quite capable of figuring out appropriate data distributions (they'll calculate where the mean and standard deviations need to be on their own) but setting them at the start can help our networks achieve better performance quicker.

# transform = torchvision.transforms.Compose([
#     transforms.Resize(size=(224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
# ])

# 2. Automatic Creation
    # When you setup a model from torchvision.models and select the pretrained model weights you'd like to use, for example, say we'd like to use:
        # weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        # Where, `EfficientNet_B0_Weights`` is the model architecture weights we'd like to use (there are many differnt model architecture options in torchvision.models).
        # DEFAULT means the best available weights (the best performance in ImageNet).
        # Note: Depending on the model architecture you choose, you may also see other options such as IMAGENET_V1 and IMAGENET_V2 where generally the higher version number the better. Though if you want the best available, DEFAULT is the easiest option. See the torchvision.models documentation for more.

# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transforms = weights.transforms()

print(f"Using {weights} weights for our EfficientNetB0 model.") # EfficientNet_B0_Weights.IMAGENET1K_V1
print(f"Transforms used by the model: {auto_transforms}")

train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=auto_transforms, batch_size=BATCH_SIZE, num_workers=0) # TODO: look into why any num_workers > 0 causes an error
print(f"There are {len(class_names)} classes: {class_names}")

# ============================================
# 2. Getting a Pretrained Model
# ============================================
# Generally, the higher number in the model name (e.g. efficientnet_b0() -> efficientnet_b1() -> efficientnet_b7()) means better performance but a larger model.
# You might think better performance is always better, right? That's true but some better performing models are too big for some devices. (for example trying to run a model on a mobile phone with 2GB of RAM).
# But if you've got unlimited compute power, as The Bitter Lesson states, you'd likely take the biggest, most compute hungry model you can.
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

print("\nBefore Freeze:")
summary(model=model,
        input_size=(BATCH_SIZE, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

# The model contains 3 main parts:
    # 1. features -  A collection of convolutional layers and other various activation layers to learn a base representation of vision data
    # 2. avgpool - Takes the average of the output of the features layer(s) and turns it into a feature vector.
    # 3. classifier - Turns the feature vector into a vector with the same dimensionality as the number of required output classes (since efficientnet_b0 is pretrained on ImageNet and because ImageNet has 1000 classes, out_features=1000 is the default).
# For reference, our model from previous sections, TinyVGG had 8,083 parameters vs. 5,288,548 parameters for efficientnet_b0, an increase of ~654x!

# ============================================
# 3. Freezing Parameters
# ============================================
# The process of transfer learning usually goes:
    # freeze some base layers of a pretrained model (typically the features section) 
    # adjust the output layers (also called head/classifier layers) to suit your needs.

# We can freeze all of the layers/parameters in the features section by setting the attribute requires_grad=False.
for param in model.features.parameters():
    param.requires_grad = False

# Adjust output layer to our desired number of classes (3 in our case)
    # Note: Dropout layers randomly remove connections between two neural network layers with a probability of p. For example, if p=0.2, 20% of connections between neural network layers will be removed at random each pass. This practice is meant to help regularize (prevent overfitting) a model by making sure the connections that remain learn features to compensate for the removal of the other connections (hopefully these remaining features are more general).
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=len(class_names))
).to(device)

if os.path.exists(f"models/{MODEL_NAME}.pth"):
    model.load_state_dict(torch.load(f"models/{MODEL_NAME}.pth"))

print("\nAfter Freeze:")
summary(model=model,
        input_size=(BATCH_SIZE, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

# Note: The more trainable parameters a model has, the more compute power/longer it takes to train.
# Freezing the base layers of our model and leaving it with less trainable parameters means our model should train quite quickly.
# This is one huge benefit of transfer learning, taking the already learned parameters of a model trained on a problem similar to yours and only tweaking the outputs slightly to suit your problem.

# ============================================
# 4. Training
# ============================================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

eval_results = []

res = train_model(model=model,
                train_data=train_dataloader,
                test_data=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                name=MODEL_NAME,
                epochs=100,
                backups_during_training=False)

eval_results.append(res["eval_results"])
print_eval_results_table(eval_results)
save_best_model(eval_results=eval_results, model_name=MODEL_NAME, models_dir="models", results_dir="data/pizza_steak_sushi_20_percent")
plot_loss_curves(res["epoch_results"])
plt.show()

