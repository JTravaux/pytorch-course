import torch
from torch import nn, optim
from models import BlobModel
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary, show_side_by_side_decision_boundary, accuracy_fn
from torchmetrics import Accuracy

# ===============================
# Binary Classification = one thing or another (cat or dog, spam or not spam, fraud or not fraud, etc.)
# Multi-class Classification = one thing out of many (cat, dog, bird, etc.)
# ===============================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 1. Create dataset
# ===============================
NUM_CLASSES = 4 
NUM_FEATURES = 2 
RANDOM_SEED = 42

X, y = make_blobs(n_samples=1000,
                  n_features=NUM_FEATURES,
                  centers=NUM_CLASSES,
                  cluster_std=1.5, # give the clusters a little shake up 
                  random_state=RANDOM_SEED)

# Plot dataset
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.show()

# Turn data into tensors and move to device
X = torch.from_numpy(X).float().to(device)
y = torch.from_numpy(y).long().to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)

print(f"X_train first 5 rows:\n{X_train[:5]}")
print(f"y_train first 5 rows:\n{y_train[:5]}")

# ===============================
# 2. Create model
# ===============================
model = BlobModel(input_features=NUM_FEATURES, output_features=NUM_CLASSES).to(device)
print(model)

# Loss fn and optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Getting prediction probabilities for a multi-class classification problem
model.eval()
with torch.inference_mode():
    y_logits = model(X_test) 
    print(f"y_probs first 5 rows:\n{y_logits[:5]}")

# In order to evaluate and train our model, we need to convert our logits into probabilities then into labels (logits -> probabilities -> labels)
# We can do this by using the softmax function (which returns the probability of each class, equal to 1)
# However, this can also be done in the model by adding a softmax layer at the end of the model
y_probs = torch.softmax(y_logits, dim=1)
print(f"y_probs first 5 rows:\n{y_probs[:5]}")

# # We can get the predicted labels by using the argmax function (which returns the index of the maximum value)
y_pred = torch.argmax(y_probs, dim=1)
print(f"y_pred first 5 rows:\n{y_pred[:5]}")

# ===============================
# 3. Train & test loop
# ===============================
NUM_EPOCHS = 1000

for epoch in range(NUM_EPOCHS):
    model.train()

    # Forward pass
    y_logits = model(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    # Calculate loss
    loss_value = loss(y_logits, y_train)
    acc = accuracy_fn(y_pred=y_pred, y_true=y_train)

    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

    # Test
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_loss = loss(test_logits, y_test)

        y_pred_test = torch.softmax(test_logits, dim=1).argmax(dim=1)
        acc_test = accuracy_fn(y_pred=y_pred_test, y_true=y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss_value:.3f}, Accuracy: {acc:.3f}, Test Loss: {test_loss:.3f}, Test Accuracy: {acc_test:.3f}")

# ====================================
# 4. Make predictions & evaluate model
# ====================================
show_side_by_side_decision_boundary(model, X_train, y_train, X_test, y_test)

# More ways to evaluate our model (https://www.learnpytorch.io/02_pytorch_classification/#9-more-classification-evaluation-metrics)
    # 1. Accuracy - Out of 100 predictions, how many did we get right?
    # 2. Precision
    # 3. Recall
    # 4. F1 Score (harmonic mean of precision and recall)
    # 5. Confusion Matrix
    # 6. Classification Report

print(classification_report(y_true=y_test.cpu(), y_pred=y_pred_test.cpu()))

# ===============================
# 5. Save & load model
# ===============================
torch.save(model.state_dict(), "models/02_blob_model.pth")

