import torch
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
from torchmetrics.functional import accuracy
from sklearn.model_selection import train_test_split
from helper_functions import show_side_by_side_decision_boundary, accuracy_fn

RANDOM_SEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create Data (from: https://cs231n.github.io/neural-networks-case-study/)
N = 500 # number of points per class
D = 2 # dimensionality
K = 4 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
  
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.show()

# Turn data into tensors
X = torch.from_numpy(X).float().to(device)
y = torch.from_numpy(y).long().to(device)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Define model
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 64),
    nn.Tanh(),
    nn.Linear(64, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 16),
    nn.Tanh(),
    nn.Linear(16, K),
).to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    model.train()

    y_logits = model(X_train)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_test_logits = model(X_test)
        y_test_preds = torch.softmax(y_test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(y_test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=y_test_preds)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Train Loss: {loss:.4f}, Train Acc: {acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Plot decision boundary
show_side_by_side_decision_boundary(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Save model
torch.save(model.state_dict(), "models/02_exercise_3_model_2.pth")
