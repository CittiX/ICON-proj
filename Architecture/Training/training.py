import numpy as np
import torch
import torch.nn as nn
from training_configuration import training_conf

from Architecture.net_arch import in_tensor, out_tensor, optimizer, model


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


# Domain over which NN will be evaluated
dom_tensor = gen_eval_domain(30, 100)

# Train the NN and record predictions over the whole domain for visualization
eval_pred = train_and_record(model, optimizer, in_tensor, out_tensor, dom_tensor, training_conf["epochs"], training_conf["anim_record_freq"])