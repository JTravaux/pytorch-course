import torch
from torch import nn, optim
import pandas as pd
import matplotlib.pyplot as plt
from models import CircleModelV1
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary, plot_predictions

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# ==============================
# 1. Data
# ==============================

n_samples = 1000
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

# Make DataFrame of circle data
circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
print("First 5 rows of circle data:")
print(circles.head())

# Plot circles
# plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show(block=False)

# Check the input and output shapes
print(f"\nShape of X before converting: {X.shape}")
print(f"Shape of y before converting: {y.shape}")

# Turn data into tensors and move to device
X = torch.from_numpy(X).type(torch.float32).to(device)
y = torch.from_numpy(y).type(torch.float32).to(device)

# Check the input and output shapes
print(f"Shape of X after converting: {X.shape}")
print(f"Shape of y after converting: {y.shape}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nNumber of samples in training set: {len(X_train)}")
print(f"Number of samples in test set: {len(X_test)}")

# ==============================
# 2. Model
# ==============================

# model_v1 = CircleModelV1().to(device)
# print(model_v1)

model_v1 = nn.Sequential(
    nn.Linear(in_features=2, out_features=8),
    nn.Linear(in_features=8, out_features=1),
).to(device)

print(f"\nmodel_v1 state_dict: {model_v1.state_dict()}")

# Make predictions with the untrained model
# with torch.inference_mode():
#     untrained_preds = model_v1(X_test.to(device))

# print(f"Length of predictions before training: {len(untrained_preds)} | Shape of predictions before training: {untrained_preds.shape}")
# print(f"Length of test labels: {len(y_test)} | Shape of test labels: {y_test.shape}")
# print(f"\nFirst 10 predictions before training: {torch.round(untrained_preds[:10])}")
# print(f"\nFirst 10 test labels: {y_test[:10]}")

# Loss function and optimizer
# loss_fn = nn.BCELoss() # Requires the sigmoid activation function to be applied to the output layer
loss_fn = nn.BCEWithLogitsLoss() # Has the sigmoid activation function built in

optimizer = optim.SGD(params=model_v1.parameters(), lr=0.01)

# Calculate accuracy - out of 100 example, what percentage did the model get correct?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_true)) * 100

# ==============================
# 3. Training
# ==============================

# In short, the steps for training are:
    # 1. Forward pass: compute prediction
    # 2. Calculate loss
    # 3. Optimizer zero grad
    # 4. Loss backward (backpropagation)
    # 5. Optimizer step (gradient descent)

# However, currently we are missing a few steps:
    # 1. What about these "logits" (BCEWithLogitsLoss)? (logits are the "raw predictions" outputted by the model)
        # We need to transform those logits into probabilities and then into labels (raw logits -> prediction probabilities -> prediction labels)
        # 1a) We can convert these logits into probabilities by passing them through the "sigmoid" -activation function- ("softmax" for multiclass classification)
        # 1b) We can convert those probabilities into labels by rounding for binary classifications (or by taking the argmax of the probabilities for multiclass classification)

# View the first 5 outputs of the forward pass on the test data
model_v1.eval()
with torch.inference_mode():
    y_logits = model_v1(X_test) # Raw output of the model (without going through an activation functions)

print(f"\n1a) First 5 logits: {y_logits[:5].squeeze()}")

# ...as we can see from above, the logits are not in the same format as the labels
# Use the sigmoid activation function to convert the logits into probabilities (https://www.learnpytorch.io/02_pytorch_classification/#0-architecture-of-a-classification-neural-network)
    # For multiclass classification, we would use the softmax activation function
y_pred_probs = torch.sigmoid(y_logits)
print(f"1b) First 5 prediction probabilities: {y_pred_probs[:5].squeeze()}")

# For out predictions, we need to perform a range-style rounding on them (>= 0.5 = class 1, < 0.5 = class 0) - Get the labels from the probabilities
y_preds = torch.round(y_pred_probs)
print(f"1c) First 5 predictions: {y_preds[:5].squeeze()}")
print(f"\nFirst 5 test labels: {y_test[:5].squeeze()}\n")

# The training and testing loop
EPOCHS = 3000

for epoch in range(EPOCHS):
    model_v1.train()

    # Forward pass
    y_logits = model_v1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # Turn logits into probabilities and then into labels

    # Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train) # Since we are using BCEWithLogitsLoss, we can pass in the logits directly. If we were using BCELoss, we would need to pass in the probabilities: loss = loss_fn(torch.sigmoid(y_logits), y_train)
    acc = accuracy_fn(y_train, y_pred)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Backward pass (backpropagation)
    loss.backward()

    # Optimizer step (gradient descent)
    optimizer.step()

    # Test loop
    model_v1.eval()
    with torch.inference_mode():
        test_logits = model_v1(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_preds)

    # Print metrics
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch + 1} | Loss: {loss:.5f} | Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")

# ==============================
# 4. Make predictions with the trained model
# ==============================

# Looks like our model is not learning anything, so to inspect it let's make predictions with the trained model and visualize the results
model_v1.eval()
with torch.inference_mode():
    trained_preds = model_v1(X_test)

# Plot decision boundary of the trained model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1) # (rows, columns, index)
plt.title("Train")
plot_decision_boundary(model_v1, X_train, y_train)
plt.subplot(1, 2, 2) # (rows, columns, index)
plt.title("Test")
plot_decision_boundary(model_v1, X_test, y_test)
plt.show()

# It is just drawing a straight line through the middle of the data, which is not what we want

# ==============================
# 5. Improving the model through experimentation (hyperparameter tuning)
# ==============================
    # Here are some things we can try:
        # 1. Add more layers - Give the model more capacity to learn patterns in the data
        # 2. Increase the number of hidden units ( go from 8 to... 10? 20?)
        # 3. Fit for longer - More epochs
        # 4. Change the activation functions
        # 5. Change the learning rate
        # 6. Change the loss function
        # 7. Change the optimizer
        # 8. Change the batch size

# Generally, when doing machine learning experiments, you just want to change 1 thing at a time and record the results
# This is also called experiment tracking

# ================================================
# Continued in 01_binary_classification.py
# ================================================


