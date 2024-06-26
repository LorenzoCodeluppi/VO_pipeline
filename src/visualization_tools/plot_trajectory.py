import matplotlib.pyplot as plt
import numpy as np
from structures import Dataset
from utils.utility_tools import load_Kitti_GT

def plot_trajectory(ax, trajectory,frame_idx,ground_truth = None):
    
    ax.set_title('Global Trajectory')

    if ground_truth is not None:
     
        ground_truth_x = ground_truth[:frame_idx, 0]
        ground_truth_z = ground_truth[:frame_idx, 1]
       
        # Plot ground truth trajectory
        ax.plot(ground_truth_x, ground_truth_z, marker='o', markersize=1, color="blue", label="Ground Truth")

    # Extract x and z coordinates from the trajectory
    x = trajectory[:, 0]
    z = trajectory[:, 2]

    
    # Plot estimated trajectory
    ax.plot(x, z, marker='o', markersize=1, color="black")

def plot_local_trajectory(ax, trajectory, landmarks = None, restrict_view = True):
    
    ax.set_title('Local Trajectory')

    # Extract x and z coordinates from the trajectory
    x = trajectory[-20:, 0]
    z = trajectory[-20:, 2]

    if landmarks is not None:
        ax.scatter(landmarks[0,:], landmarks[2,:], s=10, c='black', marker='x')

    if restrict_view:
        offset = 10
        x_min, x_max = min(x), max(x)
        z_min, z_max = min(z), max(z)
        ax.set_xlim(-offset + x_min, x_max + offset)
        ax.set_ylim(-offset + z_min, z_max + offset)
    
    ax.plot(x, z, marker='x', markersize=3, color="blue", scalex= False, scaley = False)

def plot_cand(ax, state, keypoint_history, candidates_history):
    # Add a subplot for the number of candidates and keypoints
    ax.set_title('Number of Keypoints')
    ax.set_xlabel('Frame')
    ax.legend(['Keypoints'])

    ax.plot(keypoint_history[-20:], marker='x', markersize = 2, color='green')


def plot_image(ax, image, keypoints, candidates, no_keypoints= False):
    ax.imshow(image, cmap="gray")
    ax.set_xlim(left=0, right=image.shape[1])
    ax.set_ylim(top=0, bottom=image.shape[0])
    print(image.shape)
    if not no_keypoints:
        ax.scatter(candidates[0,:], candidates[1,:], s=5, c='red', marker='o')
        ax.scatter(keypoints[0,:], keypoints[1,:], s=5, c='green', marker='x')

def create_plot(axis_arr, image, state, trajectory, frame_idx, keypoint_history, candidates_history,perf_boost, ground_truth = None,):
    for idx, ax in enumerate(axis_arr):
        if idx == 0:
            plot_image(ax, image, state.get_keypoints(), state.get_candidates_points(), perf_boost)
        elif idx == 1 and not perf_boost:
            plot_trajectory(ax, trajectory, frame_idx, ground_truth)
        elif idx == 2:
            plot_local_trajectory(ax, trajectory, state.get_landmarks())
        elif idx == 3 and not perf_boost:
            plot_cand(ax, state, keypoint_history, candidates_history)


def clear_plot(axis_arr):
    for ax in axis_arr:
        ax.clear()

def plot_final_comparison(trajectory, ground_truth, last_frame):

    if ground_truth is not None:
        ground_truth_x = ground_truth[:last_frame, 0]
        ground_truth_z = ground_truth[:last_frame, 1]

    x = trajectory[:, 0]
    z = trajectory[:, 2]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    fig.set_figheight(10)
    fig.set_figwidth(10)

    ax1.set_aspect('equal', adjustable='datalim')
    ax2.set_aspect('equal', adjustable='datalim')

    # Set titles
    ax1.set_title('Ground Truth')
    ax2.set_title('Estimated Trajectory')

    if ground_truth is not None:
        # Plot ground truth trajectory
        ax1.plot(ground_truth_x, ground_truth_z, marker='o', markersize=1, color="blue", label="Ground Truth")

    # Plot estimated trajectory
    ax2.plot(x, z, marker='o', markersize=1, color="black")

    plt.show()
