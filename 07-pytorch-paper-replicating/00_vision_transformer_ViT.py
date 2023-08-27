# Goal: Take a research paper from a new model architecture and replicate it in PyTorch. (images + math + text -> code)
# Source: https://arxiv.org/abs/2010.11929 (https://arxiv.org/pdf/2010.11929.pdf) (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)

# A Transformer architecture is generally considered to be any neural network that uses the attention mechanism) as its primary learning layer.
# The original Transformer architecture was designed to work on one-dimensional (1D) sequences of text.
# Similar to a how a convolutional neural network (CNN) uses convolutions as its primary learning layer.

# Like the name suggests, the Vision Transformer (ViT) architecture was designed to adapt the original Transformer architecture to vision problem(s) (classification being the first and since many others have followed).

import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from helper_functions import set_seeds, download_data, split_data, plot_loss_curves, create_dataloaders

folder_name = "pizza_steak_sushi_20_percent"

device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = download_data(source="https://github.com/JTravaux/pytorch-course/raw/main/pizza_steak_sushi_100_percent.zip", destination=folder_name)

train_dir = image_path / "train"
test_dir = image_path / "test"

# Creating the necessary transforms (refer to table 3 in the paper)
IMG_SIZE = 224
BATCH_SIZE = 64 # The paper used 4096, however, this is too large for my GPU, so I will use a smaller amount

manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=manual_transforms, batch_size=BATCH_SIZE, num_workers=0) # TODO: look into why any num_workers > 0 causes an error

# Get a batch of images and visualize a single image/label
image_batch, label_batch = next(iter(train_dataloader))
image, label = image_batch[0], label_batch[0]
print(image.shape, label)

plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
plt.title(class_names[label])
plt.axis(False);
# plt.show()
