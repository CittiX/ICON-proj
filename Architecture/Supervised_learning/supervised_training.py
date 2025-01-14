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
training_conf["epochs"] = 5000
training_conf["anim_record_freq"] = 15
training_conf["learning_rate"] = 0.005

# Initial guesses for parameters are provided
init_drag_coef = 1
init_thrust_coef = 1

# Unknown parameters
drag_coefficient = torch.tensor(init_drag_coef, dtype=torch.float, requires_grad=True)
thrust_magnitude = torch.tensor(init_thrust_coef, dtype=torch.float, requires_grad=True)
learnable_constants = [drag_coefficient, thrust_magnitude]

# Scale factors
constant_scale_factors = {"drag_coefficient": 1,
                          "thrust_magnitude": 3000}

# Reinitialize network, optimizer, input and target tensors
model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, output_dim)
)

optimizer = optim.Adam([{"params": model.parameters(), "lr": training_conf["learning_rate"]},
                        {"params": learnable_constants, "lr": training_conf["learning_rate"] / 10}]
                       )
input_tensor = torch.tensor(inputs, dtype=torch.float).view(-1, 1)
target_tensor = torch.tensor(targets, dtype=torch.float).view(-1, 2)


def train_one_epoch_supervised(model, inputs, targets, optimizer):
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
    physics_loss = compute_physics_loss(model, learnable_constants, constant_scale_factors)

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
                             training_conf["anim_record_freq"], train_one_epoch_supervised)

drag_param_pred = learnable_constants[0].item() * constant_scale_factors["drag_coefficient"]
thrust_param_pred = learnable_constants[1].item() * constant_scale_factors["thrust_magnitude"]

print(f"Predicted drag force: {drag_param_pred:.2f}\tActual drag force: {missile.drag_coefficient}")
print(f"Predicted thrust: {thrust_param_pred:.2f}\tActual thrust force: {missile.initial_thrust}")

# Animation
anim = show_animation(eval_pred, targets, training_conf)

# For displaying use jupyter notebook
