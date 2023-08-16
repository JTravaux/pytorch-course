import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
from plot import plot_predictions
from models import LinearRegressionModelV2
from model_functions import save_model, load_model

# For device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create a simple dataset using linear regression formula (y = weight * x + bias) (y=mx+b)
weight = 0.42
bias = 0.21

# Create range values
start = 0
end = 1
step = 0.015

# X and y  (features and labels)
X = torch.arange(start, end, step).unsqueeze(1) # <--- unsqueeze() adds a dimension of 1 to the tensor - without this, errors will occur
y = weight * X + bias

# Split data into training and test sets
split = int(len(X) * 0.75)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# # Visualize data
# plot_predictions(X_train, y_train, X_test, y_test, None)

# # Create a linear model by subclassing nn.Module
# torch.manual_seed(42) # <--- set random seed for reproducibility
# model_1 = LinearRegressionModelV2()
# model_1.to(device) # <--- move model to GPU if available

# # Loss and optimizer
# loss_function = nn.L1Loss() # <--- L1Loss is the mean absolute error (MAE) loss function
# optimizer = torch.optim.SGD(model_1.parameters(), lr=0.0001) # <--- Stochastic Gradient Descent (SGD) optimizer

# # Training loop
# epochs = 12750

# epoch_count = []
# loss_values = []
# test_loss_values = []

# for epoch in range(epochs):
#     model_1.train() # <--- set model to training mode

#     y_pred = model_1(X_train.to(device)) # <--- move data to GPU if available
#     loss = loss_function(y_pred, y_train.to(device)) # <--- move data to GPU if available

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Testing
#     model_1.eval() # <--- set model to evaluation mode

#     with torch.inference_mode():
#         y_test_pred = model_1(X_test.to(device))
#         test_loss = loss_function(y_test_pred, y_test.to(device))

#     # Track metrics
#     epoch_count.append(epoch)
#     loss_values.append(loss.item())
#     test_loss_values.append(test_loss.item())

#     if epoch % 100 == 0:
#         print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {loss.item():.4f} | Test loss: {test_loss.item():.4f}")

# # Plot the loss values over time
# plt.plot(epoch_count, loss_values, label="training loss")
# plt.plot(epoch_count, test_loss_values, label="test loss")
# plt.title("Training and test loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()


# # Plot predictions
# model_1.eval() # <--- set model to evaluation mode
# with torch.inference_mode():
#     y_pred = model_1(X_test.to(device))

# print(model_1.state_dict())
# plot_predictions(X_train, y_train, X_test, y_test, y_pred.detach().cpu()) # <--- move data to CPU if available

# # Save model
# save_model(model_name="01_first_pytorch_workflow_model_1_state_dict.pth", model=model_1.state_dict())



# Load the saved model
model_1_loaded = LinearRegressionModelV2()
model_1_loaded.load_state_dict(load_model(model_name="01_first_pytorch_workflow_model_1_state_dict.pth"))
model_1_loaded.to(device) # <--- move model to GPU if available
model_1_loaded.eval() # <--- set model to evaluation mode

with torch.inference_mode():
    y_pred = model_1_loaded(X_test.to(device))

print(model_1_loaded.state_dict())
