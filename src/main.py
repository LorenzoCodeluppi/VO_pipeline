import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from structures import Dataset
from bootstrap.initialization import initialization
from continous_operations.process_frame import process_frame
from visualization_tools.plot_trajectory import plot_trajectory
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
    prev_img = database_image

    # for i in range(bootstrap_frames[1] + 1, last_frame + 1):
    for i in range(bootstrap_frames[1] + 1, 35):
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

        candidates_points = state.get_candidates_points()
        keypoints = state.get_keypoints()
        landmarks = state.get_landmarks()
    
        # Update the trajectory array
        trajectory = np.vstack([trajectory, t])

        f, axarr = plt.subplots(2,1)
        f.set_figwidth(12)
        f.set_figheight(7)

        axarr[0].imshow(prev_img, cmap="gray")
        axarr[0].scatter(candidates_points[0,:], candidates_points[1,:], s=1, c='red', marker='o')
        axarr[0].scatter(keypoints[0,:], keypoints[1,:], s=1, c='green', marker='x')

        plot_trajectory(axarr[1], trajectory)

        plt.pause(0.1)
        plt.close()
        # plt.show()
        prev_img = image
    


if __name__ == "__main__":
    dataset = Dataset.KITTI
    bootstrap_frames = [0, 2]

    K, images, last_frame = load_dataset(dataset)

    frame1, frame2 = load_bootstrap_images(dataset, bootstrap_frames, images)

    # TODO: check the implementation
    state = initialization(frame1, frame2, K)

    run_pipeline(dataset, state, bootstrap_frames, last_frame, frame2, images, K)

    