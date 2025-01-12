import numpy as np

from Architecture.Training.training import gen_eval_domain
from Architecture.net_arch import model
from Architecture.Simulation.missile_sim import missile_conf
import torch

def compute_physics_loss(model, conf=missile_conf, physics_weight=1):
    # input_tensor is time
    dom_tensor = gen_eval_domain(30, 300).requires_grad_(True)

    # Take (x, y) predictions from NN
    phys_pred = model(dom_tensor)
    x_pred = phys_pred[:, 0]
    y_pred = phys_pred[:, 1]

    # Compute first and second order NN derivatives with respect to time using autograd
    dx_dt = torch.autograd.grad(x_pred, dom_tensor, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y_pred, dom_tensor, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    d2x_dt = torch.autograd.grad(dx_dt, dom_tensor, grad_outputs=torch.ones_like(dx_dt), create_graph=True)[0]
    d2y_dt = torch.autograd.grad(dy_dt, dom_tensor, grad_outputs=torch.ones_like(dy_dt), create_graph=True)[0]

    # Mean squared loss term for initial position and initial speed, both should be [0, 0]
    initial_pos_loss = x_pred[0] ** 2 + y_pred[0] ** 2
    initial_speed_loss = dx_dt[0] ** 2 + dy_dt[0] ** 2

    # Generate physics constants and convert them to tensors
    gravity = torch.tensor(conf["gravity"], dtype=torch.float, requires_grad=False)
    launch_angle = torch.tensor(conf["launch_angle"], dtype=torch.float, requires_grad=False)
    drag_coefficient = torch.tensor(conf["drag_coefficient"], dtype=torch.float, requires_grad=False)
    thrust_magnitude = torch.tensor(conf["initial_thrust"], dtype=torch.float, requires_grad=False)
    mass = torch.tensor(conf["mass"], dtype=torch.float, requires_grad=False)

    def compute_thrust(t, thrust_cease=conf["thrust_duration"] + 1, sharpness=200, offset=0.02):
        """
        This is a reworked version of training.py function to be better adapted to torch libray
        :param t: Time
        :param thrust_cease: The time at which thrust ceases
        :param sharpness: Constant that monitors the step sharpness
        :param offset: Position corrector for the step
        :return: A differentiable step function
        """

        return 1 / (1 + np.exp(-sharpness * (thrust_cease - t + offset)))

    # Compute thrust for time inputs
    thrust_term = thrust_magnitude * compute_thrust(dom_tensor)

    # Calculate the final physics loss function starting with x-axis and y-axis
    x_axis = (thrust_term * np.cos(launch_angle) - drag_coefficient * dx_dt) / mass
    y_axis = (thrust_term * np.sin(launch_angle) - drag_coefficient * dy_dt - mass * gravity) / mass

    physics_loss = ((d2x_dt - x_axis) ** 2 + (d2y_dt - y_axis) ** 2).mean()

    # Add initial position losses and return total loss multiplied by lambda weight
    return physics_weight * (physics_loss + initial_pos_loss + initial_speed_loss)