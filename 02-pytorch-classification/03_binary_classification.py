import torch
from torch import nn, optim
import pandas as pd
import matplotlib.pyplot as plt
from models import CircleModelV3
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary, accuracy_fn

# ==============================
# For this attempt, we introduce non-linearities to the model
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# ==============================
# 1. Data
# ==============================
X, y = make_circles(n_samples=1000,
                    noise=0.03,
                    random_state=42)

# Make DataFrame of circle data
circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
print("First 5 rows of circle data:")
print(circles.head())

# Turn data into tensors and move to device
X = torch.from_numpy(X).type(torch.float32).to(device)
y = torch.from_numpy(y).type(torch.float32).to(device)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nNumber of samples in training set: {len(X_train)}")
print(f"Number of samples in test set: {len(X_test)}")

# ==============================
# 2. Model (Non-linearality)
# ==============================
model_v3 = CircleModelV3().to(device)
print(f"\nmodel_v2 state_dict: {model_v3.state_dict()}")

# Loss function and optimizer
# loss_fn = nn.BCELoss() # Requires the sigmoid activation function to be applied to the output layer
loss_fn = nn.BCEWithLogitsLoss() # Has the sigmoid activation function built in
optimizer = optim.SGD(params=model_v3.parameters(), lr=0.1)

# ==============================
# 3. Training
# ==============================

# The training and testing loop
EPOCHS = 1200

for epoch in range(EPOCHS):
    model_v3.train()

    # Forward pass
    y_logits = model_v3(X_train).squeeze() # Without the squeeze, the shape of y_logits would be (800, 1) instead of (800,)
    y_pred = torch.round(torch.sigmoid(y_logits)) # Turn logits into probabilities and then into labels

    # Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train) # Since we are using BCEWithLogitsLoss, we can pass in the logits directly. If we were using BCELoss, we would need to pass in the probabilities: loss = loss_fn(torch.sigmoid(y_logits), y_train)
    acc = accuracy_fn(y_train, y_pred) # Optional

    # Optimizer zero grad
    optimizer.zero_grad()

    # Backward pass (backpropagation)
    loss.backward()

    # Optimizer step (gradient descent)
    optimizer.step()

    # Test loop
    model_v3.eval()
    with torch.inference_mode():
        test_logits = model_v3(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_preds)

    # Print metrics
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch + 1} | Loss: {loss:.5f} | Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")

# ==============================
# 4. Make predictions with the trained model
# ==============================
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1) # (rows, columns, index)
plt.title("Train")
plot_decision_boundary(model_v3, X_train, y_train)
plt.subplot(1, 2, 2) # (rows, columns, index)
plt.title("Test")
plot_decision_boundary(model_v3, X_test, y_test)
plt.show()
