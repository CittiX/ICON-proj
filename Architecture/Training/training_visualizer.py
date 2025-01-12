import matplotlib.pyplot as plt
from matplotlib import animation

from Architecture.Simulation.dataset import noisy_targets
from Architecture.Training.training_configuration import training_conf
from Architecture.Training.training import eval_pred


def show_animation(eval_pred, targets, conf=training_conf):
    fig, ax = plt.subplots()
    ax.set_xlim(-50, 300)
    ax.set_ylim(-50, 850)

    scatter_true, = ax.plot([], [], 'o', markersize=4, label="Ground Truths", c="#fc0303")
    scatter2, = ax.plot([], [], 'x', markersize=4, label="Predicted Points", c="#4242f5")
    line, = ax.plot([], [], lw=2, label = "Neural Network Prediction", c="#f542e0", alpha=0.5)

    def init():
        scatter_true.set_data([], [])
        scatter2.set_data([], [])
        line.set_data([], [])
        return scatter_true, line

    def animate(i):
        scatter_true.set_data(targets[:, 0], targets[:, 1])
        predictions = eval_pred[i]
        dom_second_indices = [*range(0, len(predictions), 10)]
        scatter2.set_data(predictions[dom_second_indices, 0], predictions[dom_second_indices, 1])
        line.set_data(predictions[:, 0], predictions[:, 1])

        return scatter_true, line

    anim = animation.FuncAnimation(fig, animate, init_func=init, blit=True, frames=len(eval_pred), interval=conf["anim_frame_duration"])
    plt.legend(loc="lower center", fontsize="small")
    plt.title("Neural Network Training")
    plt.close(fig)

    return anim

anim = show_animation(eval_pred, noisy_targets)

# UNCOMMENT if you want the animation to be saved as .gif
# anim.save("training_animation.gif", writer="pillow", fps=10)