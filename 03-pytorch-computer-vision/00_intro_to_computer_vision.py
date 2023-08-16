# What are some uses of computer vision?
    # Binary classification (is this a picture of a dog or a cat?)
    # Multi-class classification (is this a picture of a dog, cat, or human?)
    # Object detection (is there a dog, cat, or human in this picture? If so, where are they?)
    # Segmentation (what pixels belong to the dog, cat, or human in this picture?)

# Getting a vision dataset to work with using torchvision.datasets
# Archetecture of a CNN (Convolutional Neural Network) with PyTorch
# End to end multi-class image classification with PyTorch

# Steps in modelling with CNNs in PyTorch
    # 1. Get data ready (download, transform, load)
    # 2. Create a model
    # 3. Set up loss and optimizer
    # 4. Train the model
    # 5. Evaluate the model

# What is a CNN (Convolutional Neural Network)?
    # A series of convolutional layers, pooling layers, and fully connected layers
    # Convolutional layers are responsible for finding patterns in images
    # Pooling layers are responsible for reducing the spatial dimensions of an image
    # Fully connected layers are responsible for combining patterns across an image

# torchvision.datasets - Get datasets and data loading functions for vision data
# torchvision.models - Get pre-trained models that you can leverage for your own work
# torchvision.transforms - Operations you can apply to your image dataset to transform them
# torchvision.utils.data.Dataset - Base dataset class for PyTorch that you can make your own dataset inherit from 
# torchvision.utils.data.DataLoader - Makes your image data batches iterable


import torch
import torchvision
# from torchvision import datasets, transforms, models

# Versions
print(torch.__version__)
print(torchvision.__version__)
