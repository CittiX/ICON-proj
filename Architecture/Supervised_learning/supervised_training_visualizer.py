import numpy as np
from matplotlib import animation, pyplot as plt

from Architecture.Simulation.missile_sim import missile
from Architecture.Supervised_learning.training import gen_eval_domain
from Architecture.Supervised_learning.supervised_training import model

# Initialize graphs
fig, ax = plt.subplots()
line_real, = ax.plot([], [], c="#FC03E8", lw=1.5, alpha=0.8, ls='-', label="Simulated Trajectory")
line_PINN, = ax.plot([], [], c="#f542e0",lw=1.5, alpha=0.8, ls='-', label="Predicted Trajectory")
scat_real = ax.scatter([], [], c="#fc0303", marker='o', alpha=0.8, s=25)
scat_PINN = ax.scatter([], [], c="#4242f5", marker='x', alpha=0.8, s=30)

def zero_pad(arr1, arr2):
    "Append zeros to the end of an array so that arrays are the same size for use in the animation"
    len1 = len(arr1)
    len2 = len(arr2)

    if len1 > len2:
        return arr1, np.pad(arr2, ((0, len1 - len2), (0, 0)), "edge")
    else:
        return np.pad(arr1, ((0, len2 - len1), (0, 0)), "edge"), arr2

def gen_scatter_data(y_data, step=10):
    "Visualizes points out of bounds except every 10th point, which is shown"
    scatter_data = np.full((len(y_data), 2), [-10, -10])
    scatter_data[::step] = y_data[::step]

    return scatter_data

def init():
    line_real.set_data([], [])
    line_PINN.set_data([], [])
    scat_real.set_offsets([-10, -10])
    scat_PINN.set_offsets([-10, -10])

    return line_real, line_PINN, scat_real, scat_PINN

def update(frame):
    line_real.set_data(y_missile[:frame, 0], y_missile[:frame, 1])
    line_PINN.set_data(y_model[:frame, 0], y_model[:frame, 1])
    scat_real.set_offsets(scatter_data_real[:frame])
    scat_PINN.set_offsets(scatter_data_PINN[:frame])

    return line_real, line_PINN, scat_real, scat_PINN



# Generate data for graph
x = gen_eval_domain(20, 1000)
y_missile = missile.pos_arr[::5]
y_model = model(x).detach().cpu().numpy()[::5]

# Equalize array lengths via zero padding
y_missile, y_model = zero_pad(y_missile, y_model)

# Generate scatter point data arrays
scatter_data_real = gen_scatter_data(y_missile)
scatter_data_PINN = gen_scatter_data(y_model)

# Formatting
ax.set_xlim(0, max(y_missile[:, 0].max(), y_model[:, 0].max()) + 50)
ax.set_ylim(0, max(y_missile[:, 1].max(), y_model[:, 1].max()) + 50)
ax.legend(loc="upper right")

# Create animation
anim = animation.FuncAnimation(fig, update, frames=len(y_missile) + 100, init_func=init,
                               blit=True, interval=60)
plt.title("Real VS Predicted Trajectories")
plt.close()

# Use jupyter to show the animated graph