import numpy as np
import pandas as pd

# Simulation parameters
num_samples = 1000
GRAVITY = 9.81 # m/s^2

# Starting parameters setup
np.random.seed(42)
initial_speed = np.random.uniform(200, 800, num_samples) # m/s
launch_angle = np.random.uniform(0, 360, num_samples) # degrees
wind_speed = np.random.uniform(0, 20, num_samples) # m/s
target_distance = np.random.uniform(5000, 20000, num_samples) # m
initial_thrust = np.random.uniform(2000, 5000, num_samples)

# Calculating trajectories
trajectories = []
for speed, angle, wind, target, thrust in zip(initial_speed, launch_angle, wind_speed, target_distance, initial_thrust):
    angle_rad = np.radians(angle)
    max_height = (speed ** 2) * (np.sin(angle_rad) ** 2) / (2 * GRAVITY)
    # Max range without wind
    range_nwind = (speed ** 2) * np.sin(2 * angle_rad) / GRAVITY
    # Wind adaptation
    range_wind = range_nwind - (wind * 0.1 * range_nwind)
    accuracy = np.abs(target - range_wind)
    trajectories.append((max_height, range_nwind, accuracy))


# DataSet generation
data = pd.DataFrame({
    "Initial Speed": initial_speed,
    "Launch Angle": launch_angle,
    "Initial Thrust": initial_thrust,
    "Wind Speed": wind_speed,
    "Target Distance": target_distance,
    "Max Height": [traj[0] for traj in trajectories],
    "Range (affected by Wind)": [traj[1] for traj in trajectories],
    "Accuracy": [traj[2] for traj in trajectories]
})

# To CSV
dataset_path = "./../aa_defense_dataset.csv"
data.to_csv(dataset_path, index=False)