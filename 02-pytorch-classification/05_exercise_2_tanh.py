import torch
import matplotlib.pyplot as plt

# Create stright line tensor
X = torch.arange(-10, 10, 0.1)

plt.plot(torch.tanh(X))
plt.show()

def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

plt.plot(tanh(X))
plt.show()
