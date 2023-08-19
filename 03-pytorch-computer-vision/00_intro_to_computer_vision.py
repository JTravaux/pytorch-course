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

# PyTorch
import random
import time
import torch
from torch import nn

# Torchvision
import torchvision
from torchvision import datasets, transforms
from torchmetrics import ConfusionMatrix

# Other
from mlxtend.plotting import plot_confusion_matrix
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from models import FashionMNISTModelV0, FashionMNISTModelV1, FashionMNISTModelV2
from helper_functions import eval_model, train_step, test_step, print_eval_results_table, save_best_model, make_predictions

# Versions
print(torch.__version__)
print(torchvision.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================
# 1. Get Dataset(s)
# ===================

# For this example, we will be using the Fashion-MNIST dataset (https://github.com/zalandoresearch/fashion-mnist),
# which is a more complex/newer version of the MNIST dataset (https://en.wikipedia.org/wiki/MNIST_database)
# Pytorch docs: https://pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST

train_data = datasets.FashionMNIST(
    root="data", # Where the data is stored locally
    train=True, # Is this training data?
    download=True, # If the data isn't already downloaded, download it
    transform=transforms.Compose([transforms.ToTensor()]), # Transform the data. Compose allows us to chain multiple transforms together, but in this case we only have one transform so it doesn't matter
    target_transform=None, # Transform the target (in this case, the labels)
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

print(len(train_data), len(test_data))

image, label = train_data[0]
print(f"Image shape: {image.shape}, Label: {label}")  # 1 channel (grayscale), 28x28 pixels, label is 9 (ankle boot) [color channel, height, width]
print(train_data.class_to_idx) # The classes in the dataset

# Visualize the data as an image
# plt.imshow(image.squeeze(), cmap="gray") # Matplotlib expects the color channel to be the last dimension (or omitted), so we use squeeze to remove the color channel dimension (1)
# plt.title(train_data.classes[label])
# plt.axis(False)
# plt.show()

# Visualize the data as a grid of images
# rows, cols = 4, 4
# fig = plt.figure(figsize=(cols * 2, rows * 2))

# for i in range(1, rows * cols + 1):
#     randomIndex = torch.randint(len(train_data), size=([1])).item()
#     image, label = train_data[randomIndex]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(image.squeeze(), cmap="gray")
#     plt.title(train_data.classes[label])
#     plt.axis(False)

# plt.show()

# Looks like some of the images/labels are similar, but not the same. For example, the shirt and coat look similar, but are different classes.
# Will this cause problems for our model? Can this be modelled with pure linear lines? Or will we need non-linearaities?

# Prepare the data (DataLoader)
print(type(train_data)) # <class 'torchvision.datasets.mnist.FashionMNIST'>
# Right now our data is of type FashionMNIST, but we need to convert it to a DataLoader so that we can iterate over it
# We need to separate the data into batches because eventually we will be training on thousands of images, and we can't load them all into memory at once
# We will also want to shuffle the data so that the model doesn't learn the order of the data
# Note: Mini-batch gradient descent

BATCH_SIZE = 32 # Most common batch size is 32, 64, 128, 256, 512, 1024
train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False) # Generally we don't shuffle the test data

print(f"Length of train dataloader (# batches): {len(train_dataloader)}")
print(f"Length of test dataloader (# batches): {len(test_dataloader)}")

# Visualize a batch of data (show a sample of the data)
train_images, train_labels = next(iter(train_dataloader))
print(train_images.shape, train_labels.shape) # 32 images, 1 channel, 28x28 pixels

# flatten_layer = nn.Flatten()
# sample = train_images[0]
# flat_sample = flatten_layer(sample)

# print(f"\nShape before flatten: {sample.shape}")
# print(f"Shape after flatten: {flat_sample.shape}")

# dummy_x = torch.rand([1,1,28,28]).to(device)
# model.eval()
# with torch.inference_mode():
#     dummy_out = model(dummy_x)

# print(f"Shape before model: {dummy_x.shape}")
# print(f"Shape after model: {dummy_out.shape}, Logits: {dummy_out}")

# ===================
# 2. Create Model(s)
# ===================
HIDDEN_UNITS = 64
retrain = True

live_model = FashionMNISTModelV2(input_shape=1, hidden_units=HIDDEN_UNITS, output_shape=len(train_data.classes)).to(device)
if retrain:
    live_model.load_state_dict(torch.load("models/03_fashion_mnist_model_best.pth"))

models = [
    # FashionMNISTModelV0(input_shape=28*28, hidden_units=HIDDEN_UNITS, output_shape=len(train_data.classes)),
    # FashionMNISTModelV1(input_shape=28*28, hidden_units=HIDDEN_UNITS, output_shape=len(train_data.classes)),
    # FashionMNISTModelV2(input_shape=1, hidden_units=HIDDEN_UNITS, output_shape=len(train_data.classes)),
    # FashionMNISTModelV0(input_shape=28*28, hidden_units=HIDDEN_UNITS, output_shape=len(train_data.classes)).to(device),
    # FashionMNISTModelV1(input_shape=28*28, hidden_units=HIDDEN_UNITS, output_shape=len(train_data.classes)).to(device),
    # FashionMNISTModelV2(input_shape=1, hidden_units=HIDDEN_UNITS, output_shape=len(train_data.classes)).to(device),
    live_model,
]

# Loss function (optimizer is defined later)
loss_fn = nn.CrossEntropyLoss() # Since we're doing multi-class classification, we use CrossEntropyLoss

# ===================
# 3. Train Model
# ===================
EPOCHS = 5

# The optimizer will update the model's parameters once per batch rather than once per epoch
# This is called mini-batch gradient descent

# 1. Loop through epochs
# 2. Loop through train batches, perform training steps, calculate the train loss per batch
# 3. Loop through test batches, perform evaluation steps, calculate the test loss per batch
# 4. Print out what's happening
eval_results = []
verbose = True
train_model = False

if train_model:
    print(f"Starting training of {len(models)} models...")
    for idx, model in enumerate(models):
        device = next(model.parameters()).device
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # Stochastic Gradient Descent

        print(f"\nTraining model {idx + 1}/{len(models)} ({model.__class__.__name__}) on {device}...")

        start_time = time.time()

        for epoch in range(EPOCHS):   
            print(f"Running Epoch: {epoch+1}/{EPOCHS}...")
            train_step(model=model, data_loader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device, verbose=verbose)
            test_step(model=model, data_loader=test_dataloader, loss_fn=loss_fn, device=device, verbose=verbose)

        eval_results.append(eval_model(model=model, data_loader=test_dataloader, loss_fn=loss_fn, device=device, start_time=start_time, end_time=time.time()))

    print("\n=====================")
    print("Training Complete")
    print("=====================")

    print_eval_results_table(eval_results)
    save_best_model(eval_results=eval_results, model_name="03_fashion_mnist_model_best", models_dir="models", results_dir="data/FashionMNIST")
else:
    print("Loading trained model...")
    model = FashionMNISTModelV2(input_shape=1, hidden_units=HIDDEN_UNITS, output_shape=len(train_data.classes)).to(device)
    model.load_state_dict(torch.load("models/03_fashion_mnist_model_best.pth"))
    
    # Make predictions on a batch of images
    model.eval()

    with torch.inference_mode():
        test_images, test_labels = next(iter(test_dataloader))
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        test_preds = model(test_images)

    test_samples = []
    test_labels = []

    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    plt.imshow(test_samples[0][0].squeeze(), cmap="gray")
    plt.title(test_data.classes[test_labels[0]])
    plt.axis(False)
    plt.show()

    # Make predictions on a batch of images
    pred_probs = make_predictions(model=model, data=test_samples, device=device)
    print(pred_probs[:2])

    # Get the predicted labels
    pred_labels = torch.argmax(pred_probs, dim=1)
    print(pred_labels)

    # Get the predicted class names
    pred_class_names = [test_data.classes[label] for label in pred_labels]
    print(pred_class_names)

    # Visualize the data as a grid of images
    rows, cols = 4, 4
    fig = plt.figure(figsize=(cols * 2, rows * 2))

    for i in enumerate(test_samples):
        fig.add_subplot(rows, cols, i[0] + 1)
        plt.imshow(i[1][0].squeeze(), cmap="gray")
        true_label = test_data.classes[test_labels[i[0]]]
        pred_label = pred_class_names[i[0]]

        if true_label == pred_label:
            plt.title(f"{pred_label}", color="green")
        else:
            plt.title(f"{pred_label}", color="red")
               
        plt.axis(False)

    plt.show()

    # confusion matrix: https://en.wikipedia.org/wiki/Confusion_matrix
    # 1. Make predictions on a batch of images
    # 2. Make a confusion matrix
    # 3. Visualize the confusion matrix 

    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Making predictions"):
            # Send data and targets to target device
            X, y = X.to(device), y.to(device)
            # Do the forward pass
            y_logit = model(X)
            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())

    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    print(f"y_pred_tensor: {y_pred_tensor}")

    # 2. Setup confusion matrix instance and compare predictions to targets
    confmat = ConfusionMatrix(num_classes=len(test_data.classes), task="multiclass")
    confmat_tensor = confmat(preds=y_pred_tensor,
                            target=test_data.targets)

    # 3. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
        class_names=test_data.classes, # turn the row and column labels into class names
        figsize=(10, 7)
    );

    plt.show()
        