import cv2
import numpy as np

from utils.utility_tools import calculate_avarage_depth
from structures import State
from utils import utility_tools as utils

def triangulate_points(state: State, current_R, current_t, K, triangulate_signal):
  landmarks = state.get_landmarks()
  candidates = state.get_candidates_points()
  first_obs_candidates = state.get_first_obs_candidates()
  poses = state.get_camera_pose_candidates().reshape(3,4, candidates.shape[1])
  T = poses[:,-1,:] # 3xN
  
# TRIAL WITH BEARING VECTORS FAIL
  current_camera_pose_vec = np.tile((current_t[(1,2),None]), (1, candidates.shape[1]))
  #Compute bearing vectors for current candidate point and current camera pose 
  bearing_vector = utils.bearingvector(candidates,current_camera_pose_vec)
  #Compute bearing vectors for first observation and first camera pose
  first_obs_bearing_vector = utils.bearingvector(first_obs_candidates, T[(1,2),:])

  #Compute the angle between the two bearing vectors
  angle_between = utils.angle(bearing_vector, first_obs_bearing_vector)

  poses = state.get_camera_pose_candidates()
  poses_reshaped = poses.reshape(3,4, candidates.shape[1])
  T = poses_reshaped[:,-1,:] # 3xN

  # parameters to tune
  distance_threshold = 2.2

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
    filter_mask = np.ones(poses.shape[1], dtype=bool)
 
    for pose in unique_poses.T:
      indices = np.where(np.all(poses.T == pose, axis = 1))[0]

      M1 = K @ pose.reshape(3,4)
      M2 = K @ current_pose

      selected_candidates = candidates[:, indices].astype(np.float32)
      selected_first_obs_candidates = first_obs_candidates[:, indices].astype(np.float32)
 
      points_3d_homogeneous = cv2.triangulatePoints(M1, M2, selected_first_obs_candidates, selected_candidates)
      points_3d = points_3d_homogeneous[:3,:] / points_3d_homogeneous[-1, :]
      if new_landmarks is None:
        new_landmarks = points_3d
      else:
        new_landmarks = np.concatenate((new_landmarks, points_3d), axis=1)

      filter_mask[indices] = False
      
    state.move_candidates_to_keypoints(candidates[:, ~filter_mask].astype(np.float32), new_landmarks, filter_mask)
 

