import cv2
import numpy as np

from utils.utility_tools import calculate_avarage_depth
from structures import State
from utils import utility_tools as utils

def filter_triangulated_points(points_3d, M1, M2, K, candidates, first_obs_candidates, current_t):
  distance_threshold_factor = 1.5
  # Calculate reprojection errors for each point
  projected_points_1 = cv2.projectPoints(points_3d, M1[:, :3], M1[:, 3:].flatten(), K, None)[0].reshape(-1, 2)
  projected_points_2 = cv2.projectPoints(points_3d, M2[:, :3], M2[:, 3:].flatten(), K, None)[0].reshape(-1, 2)
  # reprojection_errors = cv2.norm(projected_points_1, projected_points_2, cv2.NORM_L2) / projected_points_2.shape[0]
  reprojection_errors = np.linalg.norm(candidates.T - projected_points_1, axis=1) + np.linalg.norm(first_obs_candidates.T - projected_points_2, axis=1)

  # discard points too far from the average depth
  average_depth = np.mean(points_3d[2])
  distance_threshold = distance_threshold_factor * average_depth

  # Filter points based on the depth constraint and reprojection error threshold
  valid_points_mask = \
    (points_3d[2] > current_t[-1]) & \
    (points_3d[2] < distance_threshold) \
    # (reprojection_errors < np.mean(reprojection_errors))

  return valid_points_mask

def triangulate_points(state: State, current_R, current_t, K, triangulate_signal):
  landmarks = state.get_landmarks()
  candidates = state.get_candidates_points()
  first_obs_candidates = state.get_first_obs_candidates()
  poses = state.get_camera_pose_candidates().reshape(3,4, candidates.shape[1])
  T = poses[:,-1,:] # 3xN
  
# # TRIAL WITH BEARING VECTORS FAIL
#   current_camera_pose_vec = np.tile((current_t[(1,2),None]), (1, candidates.shape[1]))
#   #Compute bearing vectors for current candidate point and current camera pose 
#   bearing_vector = utils.bearingvector(candidates,current_camera_pose_vec)
#   #Compute bearing vectors for first observation and first camera pose
#   first_obs_bearing_vector = utils.bearingvector(first_obs_candidates, T[(1,2),:])

#   #Compute the angle between the two bearing vectors
#   angle_between = utils.angle(bearing_vector, first_obs_bearing_vector)

  poses = state.get_camera_pose_candidates()
  poses_reshaped = poses.reshape(3,4, candidates.shape[1])
  T = poses_reshaped[:,-1,:] # 3xN

  # parameters to tune
  distance_threshold = 0.5

  # calculate the distance between each poses to the current pose (t), if > than threshold select them
  distances = np.linalg.norm(T - current_t[:,None], axis=0)
  max_distance = np.max(distances)

  average_depth = calculate_avarage_depth(landmarks, current_R, current_t)
  
  if max_distance / average_depth > 0.1:
    triangulate_signal = True

  mask = distances > distance_threshold

  possible_new_landmarks = np.sum(mask)
  
  if possible_new_landmarks == 0 and triangulate_signal:
    mask = distances >= np.max(distances)

  if possible_new_landmarks > 0 or triangulate_signal:
    prev_poses = poses[:,mask]
    current_pose = np.hstack((current_R, current_t[:,None]))
    unique_poses = np.unique(prev_poses, axis = 1)
    new_landmarks = None

    filter_candidates_mask = np.ones(poses.shape[1], dtype=bool)
    filter_keypoints_mask = np.ones(poses.shape[1], dtype=bool)
 
    for pose in unique_poses.T:
      indices = np.where(np.all(poses.T == pose, axis = 1))[0]

      M1 = pose.reshape(3,4)
      M2 = current_pose

      selected_candidates = candidates[:, indices].astype(np.float32)
      selected_first_obs_candidates = first_obs_candidates[:, indices].astype(np.float32)
 
      points_3d_homogeneous = cv2.triangulatePoints(K @ M1, K @ M2, selected_first_obs_candidates, selected_candidates)
      points_3d = points_3d_homogeneous[:3,:] / points_3d_homogeneous[-1, :]

      # landmarks filter
      valid_landmark_mask = filter_triangulated_points(points_3d, M1, M2, K, selected_candidates, selected_first_obs_candidates, -(current_R @ current_t))
      points_3d = points_3d[:,valid_landmark_mask]

      if new_landmarks is None:
        new_landmarks = points_3d
      else:
        new_landmarks = np.concatenate((new_landmarks, points_3d), axis=1)

      filter_candidates_mask[indices] = False
      filter_keypoints_mask[indices[valid_landmark_mask]] = False

    if new_landmarks is not None:
      # state.move_candidates_to_keypoints(candidates[:, ~filter_candidates_mask].astype(np.float32), new_landmarks, filter_candidates_mask)
      state.move_candidates_to_keypoints(candidates[:, ~filter_keypoints_mask].astype(np.float32), new_landmarks, filter_candidates_mask)
 

