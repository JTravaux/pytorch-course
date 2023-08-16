import torch
from torch import nn, optim
import pandas as pd
import matplotlib.pyplot as plt
from models import CircleModelV2Linear
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary, plot_predictions

# ==============================
# Going back to the roots to see if our model can learn a straight line
# ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# ==============================
# 1. Data
# ==============================
weight = 0.7
bias = 0.3

start = 0
end = 1

step = 0.01

X = torch.arange(start, end, step).unsqueeze(1)
y = weight * X + bias # Linear regression formula (y = mx + b)

# Split data into train and test sets
train_split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:train_split], X[train_split:], y[:train_split], y[train_split:]
print(f"\nNumber of samples in training set: {len(X_train)}")
print(f"Number of samples in test set: {len(X_test)}")

# Plot the data
plot_predictions(X_train, y_train, X_test, y_test)
plt.show()

# Change the device to cuda if available
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# ==============================
# 2. Model
# ==============================
model_v2 = CircleModelV2Linear().to(device)
print(f"\nmodel_v2 state_dict: {model_v2.state_dict()}")

# Loss function and optimizer
loss_fn = nn.L1Loss()
optimizer = optim.SGD(params=model_v2.parameters(), lr=0.01)

# ==============================
# 3. Training
# ==============================

# The training and testing loop
EPOCHS = 7000

for epoch in range(EPOCHS):
    model_v2.train()

    y_pred = model_v2(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Test loop
    model_v2.eval()
    with torch.inference_mode():
        test_preds = model_v2(X_test)
        test_loss = loss_fn(test_preds, y_test)

    # Print metrics
    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch + 1} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

# ==============================
# 4. Make predictions with the trained model
# ==============================
model_v2.eval()
with torch.inference_mode():
    preds = model_v2(X_test)

plot_predictions(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), preds.cpu())
plt.show()

