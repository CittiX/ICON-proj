import numpy as np

from Architecture.Simulation.dataset import noisy_targets

# Dictionary containing hyperparameters for model training
training_conf = {
    "net_depth": 50,  # Number of hidden layers
    "learning_rate": 0.025,
    "epochs": 1000,
    "anim_record_freq": 3,  # The higher the number the fewer animation frames are recorded
    "anim_frame_duration": 20,  # Duration (ms) of each frame
}


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
