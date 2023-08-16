# Rough idea:
    # 1. Get data ready (create tensors)
    # 2. build or pick pretrained model
        # 2 a. Pick a loss function & optimizer
        # 2 b. Build a training loop
    # 3. Fit the model to the data and make a prediction
    # 4. Evaluate the model
    # 5. Improve through experimentation
    # 6. Save and load the trained model

import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
from pathlib import Path
from models import LinearRegressionModel
from plot import plot_predictions
from model_functions import save_model, load_model

# Preparing and Loading Data (Excel spreadsheet, images, videos, audio, dna sequences, text, etc.)
    # Create some known data using the linear regreation formula (Y = a + bX) - A stright line with known parameters
    # b = weight
    # a = bias

weight = 0.7
bias = 0.3

start = 0
end = 1

step = 0.02

X = torch.arange(start, end, step).unsqueeze(1)
y = weight * X + bias

# y = ideal output
# X = input data
# The whole point of machine learning is to find the ideal weight and bias (a.k.a. parameters) to make the ideal prediction


# Splitting data into training and test sets
    # Training set - the model learns from this data, which is typically 70-80% of the total data you have available
    # Test set - the model gets evaluated on this data to test what it has learned, this is typically 20-30% of the total data you have available
    # Validation set - the model gets evaluated on this data during training to see how it's going, this is typically 10-15% of the total data you have available
        # Don't always need a validation set

# Split data into training and test sets
split = int(len(X) * 0.8)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# print(len(X_train), len(y_train), len(X_test), len(y_test))


# Visualizing data
# plot_predictions()


# reference: http://prntscr.com/1Gp1qICymCFw
# What the model does:
    # 1. Starts with random weights and biases
    # 2. Look at training data and adjust weights and biases to make better predictions
    # 3. Repeat until model is good enough

# How?
    # 1. Gradient descent algorithm (hint: requires_grad=True)
    # 2. Backpropagation

# PyTorch model building essentials: http://prntscr.com/v9K2_LWSBFS7 / https://pytorch.org/tutorials/beginner/ptcheat.html / http://prntscr.com/hioU1OG8Dv6P
    # 1. torch.nn - a PyTorch module containing all of the building blocks, classes and functions for building neural networks
    # 2. torch.nn.Parameter - what parameters should our model learn? often a Pytorch layer from torch.nn will set these up for us
    # 3. torch.nn.Module - a PyTorch class (base class) which allows us to use object-oriented programming to define a neural network as a PyTorch module
        # 3.1. override the forward method - tells PyTorch how to go from input to output
    # 4. torch.optim - a PyTorch module for setting up optimization functions for gradient descent. We can choose from a variety of different optimization functions, such as SGD or Adam (Adam is usually a good choice)

# Create a random seed
torch.manual_seed(42)

# Checking the contents of the model
model_0 = LinearRegressionModel()
# print(list(model_0.parameters()))

# List named parameters
# print(model_0.state_dict())

# Basic predictive power without training (torch.inference_mode())
# torch.no_grad() - temporarily sets all of the requires_grad flags to false (so we don't calculate gradients), but inference_mode() is better/does more/preferred
with torch.inference_mode(): # inference mode automatically turns off gradient tracking (when we are in inference mode, we don't need to calculate gradients) - PyTorch is keeping track of less stuff, so it runs faster
    y_preds_0 = model_0(X_test)

# Check our predictions
# plot_predictions(predictions=y_preds_0) # lmao dumb model

# Things we need to train
    # A loss function (how accurate is the model's predictions? - lower is better)
    # An optimizer (how should the model update its parameters? - takes into account the loss and adjusts)
# For pytorch, we need a training loop and a testin loop

# We will be using L1 loss (mean absolute error) for this case - https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html
loss_fn = nn.L1Loss()

# Set up the optimizer (torch.optim) - https://pytorch.org/docs/stable/optim.html (in this case, we are using SGD = stochastic gradient descent)
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.0001) # lr = learning rate (how fast the model learns) - most important hyperparameter - the values come from experimentation/experience
# The higher the learning rate, the faster the model learns, but it can also overshoot the ideal parameters (larger rate = larger change, lower rate = smaller change/longer time to learn)

# Training loop
    # 0. Loop through the data
    # 1. Forward pass to make predictions (involves data moving through our model's forward() function) - Also called forward propagation
    # 2. Calculate the loss (how far off are our predictions from the truth labels?) 
    # 3. Optimizer to zero grad
    # 4. Loss backward - Move backward through the network to calculate the gradients of each of the parameters of our model with respect to the loss (**backpropagation**)
    # 5. Optimizer step - use the optimizer to update the parameters of our model to improve the loss (**gradient descent**)

# An epoch is a complete pass through the training data (hyperparameter)
epochs = 17500

# Track different metrics
epoch_count = []
loss_values = []
test_loss_values = []

print(f"Model parameters before training: {model_0.state_dict()}")

# 0. Loop through the data
for epoch in range(epochs):

    # Set the model into training mode
    model_0.train() # Default is train mode, but we can set it to eval mode if we want for testing. This sets all parameters that require gradients to True

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train) # Callculating hte models difference between the predictions (y_pred) and the truth labels (y_train)

    # =================================
    # The order of 3, 4, 5 can be changed, but this is the most common order. However, step 5 must come after backpropagation (step 4 in this case)
    # =================================

    # 3. Optimizer to zero grad
    optimizer.zero_grad() # Zero out the gradients (otherwise they will accumulate)

    # 4. Loss backward (backpropagation)
    loss.backward() # Calculate the gradients of the loss with respect to the model's parameters

    # 5. Optimizer step (gradient descent)
    optimizer.step() # Update the model's parameters based on the calculated gradients. By default, changes will accumulate, so we need to zero out the gradients (step 3)

    # =================================
    # Testing loop
    # =================================

    #  Turn off gradient tracking (we don't need to calculate gradients when we are making predictions)
    model_0.eval() # This sets all parameters that require gradients to False

    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())

        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {loss.item():.4f} | Test loss: {test_loss.item():.4f}")
        print(f"Model parameters: {model_0.state_dict()}\n")


# Plot the loss values over time
plt.plot(epoch_count, loss_values, label="training loss")
plt.plot(epoch_count, test_loss_values, label="test loss")
plt.title("Training and test loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Check the predictions
with torch.inference_mode():
    y_preds_new = model_0(X_test)

plot_predictions(X_train, y_train, X_test, y_test, y_preds_new)

# Saving and loading models
    # read: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # read: https://forums.fast.ai/t/why-is-state-dict-the-recommended-method-for-saving-pytorch-models/81709

    # 1. torch.save() - Allows you to save a PyTorch object in Python's pickle format
    # 2. torch.load() - Allows you to load a PyTorch object from a pickle file
    # 3. torch.nn.Module.load_state_dict() - Allows you to load a model's saved state dictionary (parameters)


MODEL_DICT_NAME = "01_first_pytorch_workflow_model_0_state_dict.pth" # can either be .pth or .pt
MODEL_NAME = "01_first_pytorch_workflow_model_0.pth"

# save_model(model_0, MODEL_NAME)
# save_model(model_0.state_dict(), MODEL_DICT_NAME)

# To load in a saved state dictionary, we need to create a new model instance and load the state dictionary into it
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(load_model(MODEL_DICT_NAME))
loaded_model_0.eval()

# Make predictions with the loaded model
with torch.inference_mode():
    y_preds_loaded = loaded_model_0(X_test)

# Check the predictions
plot_predictions(X_train, y_train, X_test, y_test, y_preds_loaded)
