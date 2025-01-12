import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from Architecture.Simulation.missile_sim import missile_conf
from Architecture.net_arch import in_tensor, out_tensor, optimizer, model
from training_configuration import training_conf


def train_one_epoch(model, inputs, targets, optimizer):
    """
    It trains the NN for one epoch
    :param model: Model of NN to be trained
    :param inputs: Input features
    :param targets: Target features
    :param optimizer: Optimizer
    """
    predictions = model(inputs)

    # Compute the mean squared error loss
    criterion = nn.MSELoss()
    loss = criterion(predictions, targets)  # To print loss use loss.item()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def make_eval_pred(model, domain_tensor):
    """
    This function makes predictions using a domain tensor
    :param model: Model of NN to be trained
    :param domain_tensor: Domain tensor
    :return: NN predictions across whole domain
    """
    # Avoid computing gradients as prediction making doesn't need gradients
    # This saves memory and speeds up computation
    with torch.no_grad():
        return model(domain_tensor).view(-1, 2).cpu().numpy()


def gen_eval_domain(dom_max, dom_n_points):
    eval_dom = np.linspace(0, dom_max, dom_n_points + 1)

    return torch.tensor(eval_dom, dtype=torch.float).view(-1, 1)


def train_and_record(model, optimizer, in_tensor, out_tensor, dom_tensor, epochs, freq):
    """
    It trains the NN for given number of epochs recording evaluation predictions for visualization
    :param model: Model of NN to be trained
    :param optimizer: Optimizer
    :param in_tensor: Input tensor
    :param out_tensor:  Output tensor
    :param dom_tensor: Domain tensor
    :param epochs: Number of epochs the NN will be trained
    :param freq: Frequency of the recording
    """
    eval_pred = []

    for epoch in range(epochs):
        train_one_epoch(model, in_tensor, out_tensor, optimizer)
        if epoch % freq == 0:
            eval_pred.append(make_eval_pred(model, dom_tensor))

    return eval_pred


def compute_thrust(t, thrust_cease=missile_conf["thrust_duration"] + 1, sharpness=200, offset=0.02):
    """
    It calculates thrust values using a differentiable step function (sigmoid)
    :param t: Time
    :param thrust_cease: The time at which thrust ceases
    :param sharpness: Constant that monitors the step sharpness
    :param offset: Position corrector for the step
    :return: A differentiable step function
    """
    return 1 / (1 + np.exp(-sharpness * (thrust_cease - t + offset)))


def show_computed_thrust():
    # Time values
    time_dom = np.linspace(0, 5, 5000)
    # Calculate thrust values
    thrust_curve = compute_thrust(time_dom)
    # Highlight specific points
    thrust_cease = missile_conf["thrust_duration"] + 1
    highlighted_t = np.array([thrust_cease, thrust_cease + 0.02, thrust_cease + 0.02 * 2])
    highlighted_thrust = compute_thrust(highlighted_t)

    plt.figure(figsize=(8, 4))
    plt.plot(time_dom, thrust_curve, color="#BBBBBB", label="Thrust Curve", zorder=1)
    plt.scatter(highlighted_t, highlighted_thrust, c="#444444", zorder=2)

    # Annotate highlighted points
    for t, thrust in zip(highlighted_t, highlighted_thrust):
        plt.annotate(f'({t:.2f}, {thrust:.3f})', (t, thrust), textcoords="offset points", xytext=(0, 10), ha="left")

    plt.title(r"step$_3$(Thrust)")
    plt.xlabel("Time (t)")
    plt.grid()
    plt.show()


show_computed_thrust()

# Domain over which NN will be evaluated
dom_tensor = gen_eval_domain(30, 100)

# Train the NN and record predictions over the whole domain for visualization
eval_pred = train_and_record(model, optimizer, in_tensor, out_tensor, dom_tensor, training_conf["epochs"],
                             training_conf["anim_record_freq"])
