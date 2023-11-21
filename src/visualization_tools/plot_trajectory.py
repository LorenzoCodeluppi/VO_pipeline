import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(ax, trajectory, landmarks = None):
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Camera Trajectory (X-Z plane)')

    # Extract x and z coordinates from the trajectory
    x = trajectory[:, 0]
    z = trajectory[:, 2]

    if landmarks is not None:
        ax.scatter(landmarks[0,:], landmarks[2,:], s=10, c='black', marker='x')

    plt.plot(x, z, marker='o')
