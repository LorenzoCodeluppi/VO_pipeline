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
     - current_points: np.ndarray(3, N) - 3D points in the current frame
     - prev_points: np.ndarray(3, N) - 3D points in the previous frame
     - poses: np.ndarray(3, 4, M) - Camera poses for M frames
     - current_pose: np.ndarray(3, 4) - Camera pose for the current frame
     - K: np.ndarray(3, 3) - Camera intrinsic matrix

    Output:
     - angles_degrees: np.ndarray(N,) - Angles between corresponding vectors in degrees
   """

    # Inverse camera intrinsic matrix
    K_inv = np.linalg.inv(K)

    # Extract translation and rotation matrices from poses
    T_prev = poses[:, -1, :]
    R_prev = poses[:, :3, :]

    t = current_pose[:, -1]
    R = current_pose[:, :3]

    # Homogeneous coordinates and normalization
    current_points_homo = np.vstack((current_points, np.ones((1, current_points.shape[1]))))
    prev_points_homo = np.vstack((prev_points, np.ones((1, prev_points.shape[1]))))

    normalized_current_pts = K_inv @ current_points_homo
    normalized_prev_pts = K_inv @ prev_points_homo
    
    # # Calculate bearing vectors in the camera frame
    bearing_current_cam = (R.T @ normalized_current_pts) + (R.T @ t)[:, None]
    bearing_prev_cam = np.zeros((3, current_points.shape[1]))

    for i in range(current_points.shape[1]):
      bearing_prev_cam[:, i] = (R_prev[:, :, i].T @ normalized_prev_pts[:, i]) + (R_prev[:, :, i].T @ T_prev[:, 0])

    # print("candidates", prev_points.shape)
    show_bearings(bearing_current_cam, bearing_prev_cam, t, T_prev[:, 0])
    
    # Calculate dot products and magnitudes
    dot_products = np.sum(normalized_current_pts * bearing_prev_cam, axis=0)
    magnitude_current = np.linalg.norm(normalized_current_pts, axis=0)
    magnitude_prev = np.linalg.norm(bearing_prev_cam, axis=0)

    # Calculate the cosine of the angles
    cosine_of_angles = dot_products / (magnitude_current * magnitude_prev)

    # Ensure the values are within the valid range [-1, 1]
    cosine_of_angles = np.clip(cosine_of_angles, -1.0, 1.0)

    # Calculate the angles in radians
    angles_radians = np.arccos(cosine_of_angles)

    # Convert the angles to degrees
    angles_degrees = np.degrees(angles_radians)
    # print(np.max(angles_degrees))
    return angles_degrees


def angle(v1,v2):
  """ Angle between two vectors
    Computes the angle between two vectors.

    Input: 
      - v1 np.ndarray(3,1) : first vector
      - v2 np.ndarray(3,1) : second vector

    Output: 
      - angle float : angle between the two vectors
  """
  v1 = v1 / np.linalg.norm(v1)
  v2 = v2 / np.linalg.norm(v2)
  angle = np.arccos(np.dot(v1.T,v2))
  angle = np.rad2deg(angle)
  return angle

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

  # Initialize an empty array to store poses
  all_poses = []

# Loop through each .txt file in the directory
  for file_name in sorted(os.listdir(data_folder_path)):
      if file_name.endswith(".txt"):
          file_path = os.path.join(data_folder_path, file_name)
        
          # Load poses from the current file
          poses = np.loadtxt(file_path)
          
          # Append the poses to the array
          all_poses.append(poses)

  # Concatenate all loaded poses into a single array
  all_poses_array = np.concatenate(all_poses, axis=0)
  
  # Extract the first three columns of the array
  # all_poses_array = all_poses_array[:, :3]
  return  all_poses_array


def get_landmark_treshold(points_3d, distance_threshold_factor):
  average = abs(np.mean(points_3d))
  std = abs(np.std(points_3d))
  distance_threshold = average + distance_threshold_factor * std
  return distance_threshold
