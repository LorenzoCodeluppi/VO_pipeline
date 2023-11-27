import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

import params_loader as pl
from structures import Dataset
from bootstrap.initialization import initialization
from continous_operations.process_frame import process_frame
from visualization_tools.plot_trajectory import create_plot, clear_plot
from structures import State

ROOT_DIR = Path(__file__).parent.parent

data_folder_path = str(ROOT_DIR) + "/data"
kitti_path = f"{data_folder_path}/kitti"
malaga_path = f"{data_folder_path}/malaga-urban-dataset-extract-07"
parking_path = f"{data_folder_path}/parking"

def load_dataset(dataset):
    images = None
    if dataset == Dataset.KITTI:
        # need to set kitti_path to folder containing "05" and "poses"
        assert 'kitti_path' in globals(), "kitti_path not defined"
        ground_truth = np.loadtxt(f"{kitti_path}/poses/05.txt")
        ground_truth = ground_truth[:, [8, 9]]
        last_frame = 2760
        K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                    [0, 7.188560000000e+02, 1.852157000000e+02],
                    [0, 0, 1]])
        
    elif dataset == Dataset.MALAGA:
        # Path containing the many files of Malaga 7.
        assert 'malaga_path' in globals(), "malaga_path not defined"
        images = sorted(os.listdir(f"{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images"))
        left_images = images[3::2]
        images = left_images
        last_frame = len(left_images) - 1

        K = np.array([[621.18428, 0, 404.0076],
                    [0, 621.18428, 309.05989],
                    [0, 0, 1]])

    elif dataset == Dataset.PARKING:
        # Path containing images, depths, and all...
        assert 'parking_path' in globals(), "parking_path not defined"
        last_frame = 598

        K = np.loadtxt(f"{parking_path}/K.txt", delimiter=',', usecols=[0,1,2])

        ground_truth = np.loadtxt(f"{parking_path}/poses.txt")
        ground_truth = ground_truth[:, [8, 9]]
    else:
        assert False
    
    return K, images, last_frame

def load_bootstrap_images(dataset, bootstrap_frames, images):
    if dataset == Dataset.KITTI:
        img0 = cv2.imread(f"{kitti_path}/05/image_0/{bootstrap_frames[0]:06d}.png", cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(f"{kitti_path}/05/image_0/{bootstrap_frames[1]:06d}.png", cv2.IMREAD_GRAYSCALE)
    elif dataset == Dataset.MALAGA:
        img0 = cv2.imread(f"{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{images[bootstrap_frames[0]]}", cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(f"{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{images[bootstrap_frames[1]]}", cv2.IMREAD_GRAYSCALE)
    elif dataset == Dataset.PARKING:
        img0 = cv2.imread(f"{parking_path}/images/img_{bootstrap_frames[0]:05d}.png", cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(f"{parking_path}/images/img_{bootstrap_frames[1]:05d}.png", cv2.IMREAD_GRAYSCALE)
    else:
        assert False
    
    return img0, img1

def run_pipeline(dataset, state: State, bootstrap_frames, last_frame, database_image, images, K):
    # Continuous operation
    trajectory = np.zeros((0, 3))
    keypoints_history = np.zeros((0, 1))
    candidates_history = np.zeros((0, 1))

    prev_img = database_image

    fig = plt.figure()
    fig.set_figwidth(12)
    fig.set_figheight(7)

    gs = fig.add_gridspec(2,3)
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    for i in range(bootstrap_frames[1] + 1, last_frame + 1):
    # for i in range(bootstrap_frames[1] + 1, 35):
        # print(f"\n\nProcessing frame {i}\n=====================")
        if dataset == Dataset.KITTI:
            image = cv2.imread(f"{kitti_path}/05/image_0/{i:06d}.png", cv2.IMREAD_GRAYSCALE)
        elif dataset == Dataset.MALAGA:
            image = cv2.imread(f"{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{images[i]}", cv2.IMREAD_GRAYSCALE)
        elif dataset == Dataset.PARKING:
            image = cv2.imread(f"{parking_path}/images/img_{i:05d}.png", cv2.IMREAD_GRAYSCALE)
        else:
            assert False

        t = process_frame(state, prev_img, image, K)
        # Update the trajectory array
        trajectory = np.vstack([trajectory, t])
        keypoints_history = np.vstack([keypoints_history, state.get_keypoints().shape[1]])
        candidates_history = np.vstack([candidates_history, state.get_candidates_points().shape[1]])

        create_plot([ax1, ax2, ax3, ax4], image, state, trajectory, i, keypoints_history, candidates_history)
        plt.pause(0.01)
        clear_plot([ax1, ax3, ax4])
        prev_img = image

if __name__ == "__main__":
    dataset = Dataset.PARKING
    
    pl.load_parameters(dataset)

    bootstrap_frames = pl.params["bootstrap_frames"]

    K, images, last_frame = load_dataset(dataset)

    frame1, frame2 = load_bootstrap_images(dataset, bootstrap_frames, images)

    # TODO: check the implementation
    state = initialization(frame1, frame2, K)

    run_pipeline(dataset, state, bootstrap_frames, last_frame, frame2, images, K)

    

    