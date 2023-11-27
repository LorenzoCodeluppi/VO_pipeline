import matplotlib.pyplot as plt
import numpy as np
from structures import Dataset
from utils.utility_tools import load_Kitti_GT

def plot_trajectory(ax, trajectory,dataset, frame_idx):
    
    ax.set_title('Global Trajectory')

    if dataset == Dataset.KITTI:
        ground_truth_trajectory= load_Kitti_GT()
        ground_truth = ground_truth_trajectory.reshape(-1,3,4)
        
        ground_truth_x = ground_truth[frame_idx,0,3]
        ground_truth_z = ground_truth[frame_idx,2,3]
       
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

    # if restrict_view:
    #     x_range = max(x) - min(x)
    #     z_range = max(z) - min(z)
    #     x_limit = max(x) + x_range * 10
    #     z_limit = max(z) + z_range * 10
    #     ax.set_xlim(-200, 200)
    #     ax.set_ylim(-200, 200)

    ax.plot(x, z, marker='x', markersize=3, color="blue")

def plot_cand(ax, state, keypoint_history, candidates_history):
    # Add a subplot for the number of candidates and keypoints
    ax.set_title('Number of Candidates and Keypoints')
    ax.set_xlabel('Frame')
    ax.legend(['Candidates', 'Keypoints'])

    # ax.plot(candidates_history, marker='x',markersize = 2, color='red')
    ax.plot(keypoint_history[-20:], marker='x', markersize = 2, color='green')


def plot_image(ax, image, keypoints, candidates):
    ax.imshow(image, cmap="gray")
    ax.scatter(candidates[0,:], candidates[1,:], s=1, c='red', marker='o')
    ax.scatter(keypoints[0,:], keypoints[1,:], s=1, c='green', marker='x')

def create_plot(axis_arr, image, state, trajectory, frame_idx, keypoint_history, candidates_history):
    for idx, ax in enumerate(axis_arr):
        if idx == 0:
            plot_image(ax, image, state.get_keypoints(), state.get_candidates_points())
        elif idx == 1:
            plot_trajectory(ax, trajectory,Dataset.KITTI, frame_idx)
        elif idx == 2:
            plot_local_trajectory(ax, trajectory, state.get_landmarks())
        elif idx == 3:
            plot_cand(ax, state, keypoint_history, candidates_history)


def clear_plot(axis_arr):
    for ax in axis_arr:
        ax.clear()