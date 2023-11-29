import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def cross2Matrix(x):
  """ Antisymmetric matrix corresponding to a 3-vector
    Computes the antisymmetric matrix M corresponding to a 3-vector x such
    that M*y = cross(x,y) for all 3-vectors y.

    Input: 
      - x np.ndarray(3,1) : vector

    Output: 
      - M np.ndarray(3,3) : antisymmetric matrix
  """
  M = np.array([[0,   -x[2], x[1]], 
                [x[2],  0,  -x[0]],
                [-x[1], x[0],  0]])
  return M


def show_bearings(bearing_current_cam, bearing_prev_cam, current_pose, prev_pose):
  current_pose = current_pose.flatten()

  x1 = [current_pose[0], bearing_current_cam[0,-1]]
  z1 = [current_pose[2], bearing_current_cam[2,-1]]

  x2 = [prev_pose[0], bearing_prev_cam[0, -1]]
  z2 = [prev_pose[2], bearing_prev_cam[2, -1]]

  print(current_pose)
  plt.plot(x1, z1, c='blue')
  plt.plot(x2, z2, c='red')
  plt.show()
  # plt.pause(1)
  # plt.clf()


def get_angle_bearing(current_points, prev_points, poses, current_pose, K):
    """ Calculate the angle between bearing vectors in the world frame.

    Inputs:
     - current_points: np.ndarray(2, N) - 2D points in the current frame
     - prev_points: np.ndarray(2, N) - 2D points in the previous frame
     - poses: np.ndarray(3, 4, M) - Camera poses for M frames
     - current_pose: np.ndarray(3, 4) - Camera pose for the current frame
     - K: np.ndarray(3, 3) - Camera intrinsic matrix

    Output:
     - angles_degrees: np.ndarray(N,) - Angles between corresponding vectors in degrees
   """

    num_points = prev_points.shape[1]
    K_inv = np.linalg.inv(K)

    # Get Rotation matrix of current pose
    R_current = current_pose[:3, :3]

    # Convert points to homogeneous coordinates
    current_points_homogeneous = np.vstack((current_points, np.ones((1, num_points))))
    prev_points_homogeneous = np.vstack((prev_points, np.ones((1, num_points))))

    # Get 3D points (assuming lambda = 1)
    current_points_3D = K_inv @ current_points_homogeneous
    prev_points_3D = K_inv @ prev_points_homogeneous

    angles = np.zeros(num_points)

    for i in range(num_points):
        # Get R and t matrices of previous pose
        T_prev = poses[:, :, i]
        R_prev = T_prev[:3, :3]

        # De-rotate previous point to current camera frame
        deRotation = R_prev.T @ R_current
        deRotated_prev_point_3D = deRotation @ prev_points_3D[:, i]

        # Calculate dot product and magnitudes product
        dot_product = np.dot(current_points_3D[:, i], deRotated_prev_point_3D)
        magnitudes_product = np.linalg.norm(current_points_3D[:, i]) * np.linalg.norm(deRotated_prev_point_3D)

        # Calculate cosine of angle
        cos_of_angle = dot_product / magnitudes_product

        # Ensure the values are within the valid range [-1, 1]
        cos_of_angle = np.clip(cos_of_angle, -1, 1)

        # Compute angle
        angles[i] = np.arccos(cos_of_angle)

    angles_degrees = np.degrees(angles)

    return angles_degrees


def calculate_inlier_ratio(previous_keypoints, inlier_keypoints):
  number_current_keypoints = inlier_keypoints.shape[0]
  number_previous_keypoints = previous_keypoints.shape[1]
  keypoints_ratio = number_current_keypoints / number_previous_keypoints

  return keypoints_ratio

def calculate_avarage_depth(landmarks, R, t):
  position = -np.matmul(R.T, t)
  return np.mean(landmarks[-1,:]) - position[-1]

def get_validation_mask(status, error, threshold):
  if error is not None:
    valid_keypoints_mask = error < threshold
    return np.logical_and(valid_keypoints_mask, status)
  return status

def load_Kitti_GT():

  ROOT_DIR = Path(__file__).parent.parent.parent

  data_folder_path = str(ROOT_DIR) + '/data/kitti/poses'

  all_poses_array = np.loadtxt(f"{data_folder_path}/05.txt")

  return  all_poses_array


def get_landmark_treshold(points_3d, distance_threshold_factor):
  average = abs(np.mean(points_3d))
  std = abs(np.std(points_3d))
  distance_threshold = average + distance_threshold_factor * std
  return distance_threshold
