from Simulation.datasetCreator import initial_thrust, thrust_duration, drag_coefficient, initial_speed, GRAVITY, mass, \
    launch_angle, target_distance

import numpy as np
import matplotlib.pyplot as plt

MAX_SIM_TIME = 100
speed = 0
distance = 0

def simulate():
    num_time_steps = 5000
    # Step size
    dt = MAX_SIM_TIME / num_time_steps
    # Simulation duration
    time_arr = np.linspace(0, MAX_SIM_TIME, num_time_steps)

    # ODE (Ordinary differential Equations) approximation with Euler's method
    for step in range(1, num_time_steps):
        thrust = initial_thrust if time_arr[step] < thrust_duration else 0
        drag_force = -drag_coefficient * initial_speed
        gravitational_force = np.array([0, -GRAVITY * mass])
        thrust_force = thrust * np.array([np.cos(launch_angle), np.sin(launch_angle)])

        # Compute final force and acceleration
        net_force = drag_force + gravitational_force + thrust_force
        acc = net_force / mass

        speed = initial_speed + acc * dt
        distance = target_distance + speed * dt

        # End simulation if missile collides with ground
        if distance < 0:
            distance = 0
            break


def show_trajectory():
    color = '#4FEB34' # Green

    # Draw line for missile
    ballistic_phase = plt.plot(target_distance, distance, color=color, alpha=0.5,
                               lw=2, ls="--", label="Ballistic phase")
    