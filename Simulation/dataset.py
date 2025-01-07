from missile_sim import missile as missile_obj
import numpy as np
import matplotlib.pyplot as plt

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


# Generate noisy data
noisy_targets = noisy_data(missile_obj)

# Show the graph
show_noisy_data(noisy_targets, "Noisy Data",
                "X Displacement (meters)",
                "Y Displacement (meters)")
