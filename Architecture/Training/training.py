import numpy as np
import torch
import torch.nn as nn
from Architecture.Simulation.dataset import noisy_targets
from Architecture.net_arch import in_tensor, out_tensor, optimizer, model


def pre_training(noisy_data):
    """
    This function prepares the noisy data for PINN training.
    Noisy data are treated as the target (or samples or simply output).
    :param noisy_data: Noisy data for PINN training
    :return: Input-Output pairs consisting of time and observed positions
    """
    # The output/target to be predicted
    samples = noisy_data
    n_samples = len(noisy_data)
    inputs = np.linspace(0, n_samples - 1, n_samples)

    return inputs, samples


inputs, samples = pre_training(noisy_targets)

# Dictionary containing hyperparameters for model training
training_conf = {
    "net_depth": 50,  # Number of hidden layers
    "learning_rate": 0.025,
    "batch_size": 64,  # Number of examples to split the dataset in
    "epochs": 1000,
    "anim_record_freq": 3,  # The higher the number the fewer animation frames are recorded
    "anim_frame_duration": 20,  # Duration (ms) of each frame
}


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
    loss = nn.MSELoss(predictions, targets)  # To print loss use loss.item()

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
            eval_pred.append(eval_pred(model, dom_tensor))

    return eval_pred


# Domain over which NN will be evaluated
dom_tensor = gen_eval_domain(30, 100)

# Train the NN and record predictions over the whole domain for visualization
eval_pred = train_and_record(model, optimizer, in_tensor, out_tensor, dom_tensor, training_conf["epochs"], training_conf["anim_record_freq"])