from Architecture.Simulation.missile_sim import missile as missile_obj
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def noisy_data(missile_obj, n_points=8):
    """
    Generate noisy data for the missile ballistic trajectory
    :param missile_obj: A missile object
    :param n_points: Point limit
    :return: Noisy data
    """
    # Compute indices for every second
    actual_timesteps = missile_obj.pos_arr.shape[0]
    # Indices for every second
    second_indices = np.arange(0, actual_timesteps, int(1 / missile_obj.dt))

    # Every passed second find the position and limit the points to n_points
    selected_data = missile_obj.pos_arr[second_indices][:n_points]

    # Gaussian distribution
    noise = np.random.normal(0, (1 + selected_data / 100))
    noisy_data = selected_data + noise

    return noisy_data

def show_noisy_data(data, title, x_label, y_label):
    plt.scatter(data[:, 0], data[:, 1], c="#34EBCC")
    plt.xlim(-10, 500)
    plt.ylim(-10, 350)
    plt.title(title)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.show()

def export_dataset(noisy_data, filename="dataset.csv"):
    """
    Export noisy data to a .csv file
    :param noisy_data: Data to be exported
    :param filename: Name of the output file
    :return: .csv file
    """
    # Since the dataset has two columns noisy data are being converted to DataFrame
    data_frame = pd.DataFrame(noisy_data, columns=["X Displacement", "Y Displacement"])

    # Export to .csv
    data_frame.to_csv(filename, index=False)


# Generate noisy data and extract to .csv
noisy_targets = noisy_data(missile_obj)

export_dataset(noisy_targets)

# Show the graph
show_noisy_data(noisy_targets, "Noisy Data",
                "X Displacement (meters)",
                "Y Displacement (meters)")
