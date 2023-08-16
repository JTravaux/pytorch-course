# https://www.learnpytorch.io/00_pytorch_fundamentals/#reproducibility-trying-to-take-the-random-out-of-random
# https://pytorch.org/docs/stable/notes/randomness.html
# Reproducibility: Trying to take the random out of random

import torch

# A neural network learns by:
# Random numbers -> tensor operations -> update random numbers to make them better representations of the data -> repeat

# The random numbers are called "parameters" or "weights" or "learnable parameters" or "learnable weights"
# The tensor operations are called "forward pass" or "forward propagation" or "forward"

random_tensor_a = torch.randn(3, 3)
random_tensor_b = torch.randn(3, 3)
print(random_tensor_a == random_tensor_b)

# To reduce randomness, we can set the random seed
# Set the random seed
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
random_seeded_tensor_a = torch.randn(3, 3)

torch.manual_seed(RANDOM_SEED)
random_seeded_tensor_b = torch.randn(3, 3)

print(random_seeded_tensor_a == random_seeded_tensor_b)
