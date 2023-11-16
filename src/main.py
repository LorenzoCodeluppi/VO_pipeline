import os
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).parent.parent

# Setup
ds = 0  # 0: KITTI, 1: Malaga, 2: parking

data_folder_path = str(ROOT_DIR) + "/data"
kitti_path = f"{data_folder_path}/kitti"
malaga_path = f"{data_folder_path}/malaga-urban-dataset-extract-07"
parking_path = f"{data_folder_path}/parking"

bootstrap_frames = [0, 1]

if ds == 0:
    # need to set kitti_path to folder containing "05" and "poses"
    assert 'kitti_path' in locals(), "kitti_path not defined"
    ground_truth = np.loadtxt(f"{kitti_path}/poses/05.txt")
    ground_truth = ground_truth[:, [8, 9]]
    last_frame = 2760
    K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                  [0, 7.188560000000e+02, 1.852157000000e+02],
                  [0, 0, 1]])
    
elif ds == 1:
    # Path containing the many files of Malaga 7.
    assert 'malaga_path' in locals(), "malaga_path not defined"
    images = sorted(os.listdir(f"{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images"))
    left_images = images[3::2]
    images = left_images
    last_frame = len(left_images) - 1

    K = np.array([[621.18428, 0, 404.0076],
                  [0, 621.18428, 309.05989],
                  [0, 0, 1]])

elif ds == 2:
    # Path containing images, depths, and all...
    assert 'parking_path' in locals(), "parking_path not defined"
    last_frame = 598

    K = np.loadtxt(f"{parking_path}/K.txt", delimiter=',', usecols=[0,1,2])

    ground_truth = np.loadtxt(f"{parking_path}/poses.txt")
    ground_truth = ground_truth[:, [8, 9]]

else:
    assert False

# Bootstrap
# need to set bootstrap_frames
if ds == 0:
    img0 = cv2.imread(f"{kitti_path}/05/image_0/{bootstrap_frames[0]:06d}.png", cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(f"{kitti_path}/05/image_0/{bootstrap_frames[1]:06d}.png", cv2.IMREAD_GRAYSCALE)
elif ds == 1:
    img0 = cv2.imread(f"{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{images[bootstrap_frames[0]]}", cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(f"{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{images[bootstrap_frames[1]]}", cv2.IMREAD_GRAYSCALE)
elif ds == 2:
    img0 = cv2.imread(f"{parking_path}/images/img_{bootstrap_frames[0]:05d}.png", cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(f"{parking_path}/images/img_{bootstrap_frames[1]:05d}.png", cv2.IMREAD_GRAYSCALE)
else:
    assert False

# Continuous operation
for i in range(bootstrap_frames[1] + 1, last_frame + 1):
    print(f"\n\nProcessing frame {i}\n=====================")
    if ds == 0:
        image = cv2.imread(f"{kitti_path}/05/image_0/{i:06d}.png", cv2.IMREAD_GRAYSCALE)
    elif ds == 1:
        image = cv2.imread(f"{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{images[i]}", cv2.IMREAD_GRAYSCALE)
    elif ds == 2:
        image = cv2.imread(f"{parking_path}/images/img_{i:05d}.png", cv2.IMREAD_GRAYSCALE)
    else:
        assert False

    # Makes sure that plots refresh.
    cv2.imshow("Frame", image)
    cv2.waitKey(10)

    prev_img = image

cv2.destroyAllWindows()