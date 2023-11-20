import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(ax, trajectory):
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Camera Trajectory (X-Z plane)')

    # Extract x and z coordinates from the trajectory
    x = trajectory[:, 0]
    z = trajectory[:, 2]

    # Plot the updated trajectory
    return plt.plot(x, z, marker='o')
