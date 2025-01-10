import torch
import torch.nn as nn
from Architecture.Training.training_configuration import training_conf
from Architecture.Training.training_configuration import inputs, samples as targets

# Input is time (1D) and output is (x,y) coordinates vector (2D)
input_dim = 1
output_dim = 2

# Number of neurons in the hidden layer
hidden_dim = training_conf["net_depth"]

# Sequential Model configuration
model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)

# It uses ADAM as optimizer to guarantee fast convergence
optimizer = torch.optim.Adam(model.parameters(), lr=training_conf["learning_rate"])

# Use tensors to integrate GPU computing
in_tensor = torch.tensor(inputs, dtype=torch.float).view(-1, 1)
out_tensor = torch.tensor(targets, dtype=torch.float).view(-1, 2)


