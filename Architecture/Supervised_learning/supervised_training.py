"""
This package rearrange the NN training with respect of the previously computed loss function
"""
import torch
import torch.nn as nn
import torch.optim as optim

from Architecture.Simulation.missile_sim import missile
from Architecture.Supervised_learning.physics_loss import compute_physics_loss
from Architecture.Supervised_learning.training import gen_eval_domain, train_and_record
from Architecture.Supervised_learning.training_visualizer import show_animation
from Architecture.net_arch import input_dim, hidden_dim, output_dim
from Architecture.net_arch import inputs, targets
from training_configuration import training_conf

# Arrange config settings
training_conf["epochs"] = 3000
training_conf["anim_record_freq"] = 10
training_conf["learning_rate"] = 0.005

# Reinitialize network, optimizer, input and target tensors
model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)

optimizer = optim.Adam(model.parameters(), lr=training_conf["learning_rate"])
input_tensor = torch.tensor(inputs, dtype=torch.float).view(-1, 1)
target_tensor = torch.tensor(targets, dtype=torch.float).view(-1, 2)


def train_one_epoch(model, inputs, targets, optimizer):
    """
    A remake of train_one_epoch function of training.py.
    This uses mean squared error and physics loss.
    """
    # Get predictions from NN
    predictions = model(inputs)

    # Mean squared loss
    criterion = nn.MSELoss(reduction="sum")
    mse_loss = criterion(predictions, targets)

    # Physics loss
    physics_loss = compute_physics_loss(model, missile)

    # Combined loss
    loss = mse_loss + physics_loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Domain over which NN will be evaluated
dom_tensor = gen_eval_domain(30, 300)

# Train NN and record predictions over entire domain for visualization
eval_pred = train_and_record(model, optimizer, input_tensor, target_tensor, dom_tensor,
                             training_conf["epochs"],
                             training_conf["anim_record_freq"])

# Animation
anim = show_animation(eval_pred, targets, training_conf)

# For displaying use jupyter notebook
