import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torchmetrics.functional import accuracy
from sklearn.model_selection import train_test_split
from helper_functions import show_side_by_side_decision_boundary, accuracy_fn

RANDOM_SEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# Binary classification dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=RANDOM_SEED)

# Visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

X = torch.from_numpy(X).float().to(device)
y = torch.from_numpy(y).float().to(device)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Create the model
model = nn.Sequential(
    nn.Linear(2, 24),
    nn.ReLU(),
    nn.Linear(24, 24),
    nn.ReLU(),
    nn.Linear(24, 1),
).to(device)

# Loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()

    # Forward pass (logits -> probabilities -> labels)
    y_logits = model(X_train)
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Compute the loss
    loss = loss_fn(y_logits, y_train.unsqueeze(1))
    acc = accuracy(y_pred, y_train.unsqueeze(1), task="binary")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Test the model
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_test)
        y_pred = torch.round(torch.sigmoid(y_logits))
        test_acc = accuracy(y_pred, y_test.unsqueeze(1), task="binary")

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item()}, acc: {acc.item()}, test acc: {test_acc.item()}")

# Visualize the decision boundary
show_side_by_side_decision_boundary(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Save model
torch.save(model.state_dict(), "models/02_exercise_1_model.pth")
