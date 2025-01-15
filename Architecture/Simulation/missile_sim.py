import matplotlib.pyplot as plt
import numpy as np

# Uncomment below lines if you desire random values each run or fixed ones (current: fixed)
# seed = np.random.randint(0, 10000)
# seed = 42
# np.random.seed(seed)
missile_conf = {
    'gravity': 9.81,  # m/s^2
    'drag_coefficient': 3,  # kg/s
    'launch_angle': np.radians(85),  # radians
    'initial_thrust': 5000,  # N
    'thrust_duration': 2,  # s
    'mass': 100  # kg
}


# This class models a missile launched by a SAM
class Missile:
    # IDEs can't recognize attributes dynamically assigned to classes,
    # therefore missile_conf keys will eventually turnt into attributes of this class
    # and treated as such
    def __init__(self, conf):
        for key, value in conf.items():
            setattr(self, key, value)

        self.max_sim_time = 100
        self.num_time_steps = 5000
        self.dt = self.max_sim_time / self.num_time_steps  # Step size

        # Compute starting arrays for position (x, y) and speed (dx, dy)
        self.pos_arr = np.zeros((self.num_time_steps, 2))
        self.speed_arr = np.zeros((self.num_time_steps, 2))
        self.initialize()

    def initialize(self):
        # Set starting position and speed
        self.pos_arr[0, :] = 0
        self.speed_arr[0, :] = 0

    def simulate(self):
        """
        Builds a simulation for a missile instance.
        Ordinary Differential Equations are solved by Euler's method.
        :return:
        """
        # Calculate simulation duration
        time_arr = np.linspace(0, self.max_sim_time, self.num_time_steps)

        # Euler's method application
        for step in range(1, self.num_time_steps):
            # Compute forces that alters the projectile trajectory
            thrust = self.initial_thrust if time_arr[step] < self.thrust_duration else 0
            drag_force = -self.drag_coefficient * self.speed_arr[step - 1, :]
            gravitational_force = np.array([0, -self.gravity * self.mass])
            thrust_force = thrust * np.array([np.cos(self.launch_angle), np.sin(self.launch_angle)])

            # Compute force and acceleration
            net_force = drag_force + gravitational_force + thrust_force
            acc = net_force / self.mass

            # Sum acceleration to speed and then speed to position each (time) step
            self.speed_arr[step, :] = self.speed_arr[step - 1, :] + acc * self.dt
            self.pos_arr[step, :] = self.pos_arr[step - 1, :] + self.speed_arr[step, :] * self.dt

            # End simulation if missile hits the ground interpolating the ground impact point
            if self.pos_arr[step, 1] < 0:
                self.pos_arr = self.pos_arr[:step, :]
                break

    def show_trajectory(self):
        """
        Shows the trajectory of the simulation.
        :param self:
        :return:
        """
        color = "#FC03E8"

        # Line plot for missile
        ballistic_graph, = plt.plot(self.pos_arr[:, 0],
                                    self.pos_arr[:, 1],
                                    c=color,
                                    alpha=0.6,
                                    lw=2,
                                    ls="-.",
                                    label="Ballistic trajectory")

        # Show the boost phase
        boost_end_index = int(self.thrust_duration / self.dt)
        boost_graph, = plt.plot(self.pos_arr[:boost_end_index, 0],
                                self.pos_arr[:boost_end_index, 1],
                                c=color,
                                lw=2,
                                label="Boost trajectory")

        # Show every passed second as a 'scatter point'
        actual_timesteps = self.pos_arr.shape[0]
        second_indices = np.arange(0, actual_timesteps, int(1 / self.dt))
        plt.scatter(self.pos_arr[second_indices, 0],
                    self.pos_arr[second_indices, 1],
                    c=color,
                    alpha=0.9)

        # Customizing the graph plot
        plt.xlabel("X Displacement (meters)", fontsize=10)
        plt.ylabel("Y Displacement (meters)", fontsize=10)
        plt.title("Missile trajectory\n", fontsize=14)
        plt.legend(handles=[boost_graph, ballistic_graph],
                   loc="lower center",
                   fontsize=8)
        plt.grid(alpha=0.3)
        plt.show()


# Start simulation and show the graph
missile = Missile(missile_conf)
missile.simulate()
missile.show_trajectory()
