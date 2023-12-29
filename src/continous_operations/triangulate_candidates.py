import cv2
import numpy as np

import params_loader as pl
from utils.utility_tools import calculate_avarage_depth, get_landmark_treshold
from structures import State


def filter_triangulated_points(points_3d, M1, M2, K, candidates, first_obs_candidates, current_R, current_t):
  distance_threshold_factor = 2
  # Calculate reprojection errors for each point
  projected_points_1 = cv2.projectPoints(points_3d, M1[:, :3], M1[:, 3:].flatten(), K, None)[0].reshape(-1, 2)
  projected_points_2 = cv2.projectPoints(points_3d, M2[:, :3], M2[:, 3:].flatten(), K, None)[0].reshape(-1, 2)
  # reprojection_errors = cv2.norm(projected_points_1, projected_points_2, cv2.NORM_L2) / projected_points_2.shape[0]
  reprojection_errors = np.linalg.norm(candidates.T - projected_points_1, axis=1) + np.linalg.norm(first_obs_candidates.T - projected_points_2, axis=1)

  # Transform points to the camera coordinate system
  #points_3d_camera_frame = current_R @ (points_3d + current_t[:, None])
  #print("OLD implementation points to camera frame: ", points_3d_camera_frame)
  points_3d_camera_frame = current_R @ points_3d + current_t[:, None]
  #print("NEW implementation points to camera frame: ", points_3d_camera_frame)

  treshold_x = get_landmark_treshold(points_3d_camera_frame[0], distance_threshold_factor)
  treshold_z = get_landmark_treshold(points_3d_camera_frame[2], distance_threshold_factor)

  valid_points_mask = \
    (points_3d_camera_frame[0] < treshold_x) & \
    (points_3d_camera_frame[2] > 0) & \
    (points_3d_camera_frame[2] < np.min((treshold_z, 1000))) \
   
  return valid_points_mask

def triangulate_points(state: State, current_R, current_t, K, triangulate_signal):
  # parameters to tune
  angle_treshold = 30

  current_pose = np.hstack((current_R, current_t[:,None]))

  landmarks = state.get_landmarks()
  candidates = state.get_candidates_points()
  first_obs_candidates = state.get_first_obs_candidates()
  poses = state.get_camera_pose_candidates()
  poses_reshaped = poses.reshape(3,4, candidates.shape[1])
  T = poses_reshaped[:,-1,:] # 3xN

  # parameters to tune
  distance_threshold = pl.params["distance_threshold"]
  thumb_rule = pl.params["thumb_rule"]

  # calculate the distance between each poses to the current pose (t), if > than threshold select them
  distances = np.linalg.norm(T - current_t[:,None], axis=0)
  max_distance = np.max(distances)

  '''average_depth = calculate_avarage_depth(landmarks, current_R, current_t)
  print("average depth of landmarks:")
  print(average_depth)'''

  # my_avg = np.mean((current_R @ landmarks + current_t.reshape((current_t.shape[0], 1)))[2, :])
  my_avg = np.mean(landmarks[-1,:])
  print("average depth of landmarks:")
  print(my_avg)

  
  # if max_distance / my_avg > thumb_rule:
  #   triangulate_signal = True

  # mask = np.logical_or(distances > distance_threshold, angles > angle_treshold)
  # print(np.sum(angles > angle_treshold))
  mask = distances > distance_threshold
  # distances all the same after many frames??? means first obs of candidates is the same frame for all candidates
  # print(distances)
  possible_new_landmarks = np.sum(mask)
  print("mean distance", np.mean(distances), np.max(distances))
  if possible_new_landmarks == 0 and triangulate_signal:
    mask = distances >= np.mean(distances)

  if possible_new_landmarks > 0 or triangulate_signal:
    prev_poses = poses[:,mask]
    
    unique_poses = np.unique(prev_poses, axis = 1)
    print("unique poses: ", unique_poses.shape)
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
      valid_landmark_mask = filter_triangulated_points(points_3d, M1, M2, K, selected_candidates, selected_first_obs_candidates, current_R, current_t)
      points_3d = points_3d[:,valid_landmark_mask]

      print("new landmarks: ", points_3d.shape)

      if new_landmarks is None:
        new_landmarks = points_3d
      else:
        new_landmarks = np.concatenate((new_landmarks, points_3d), axis=1)

      filter_candidates_mask[indices] = False
      filter_keypoints_mask[indices[valid_landmark_mask]] = False

    if new_landmarks is not None:
      # state.move_candidates_to_keypoints(candidates[:, ~filter_candidates_mask].astype(np.float32), new_landmarks, filter_candidates_mask)
      state.move_candidates_to_keypoints(candidates[:, ~filter_keypoints_mask].astype(np.float32), new_landmarks, filter_candidates_mask)
 

