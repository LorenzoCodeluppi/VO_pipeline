import numpy as np

import params_loader as pl
from .asociation import keypoint_association
from .estimate_pose import estimate_pose
from .evaluate_points import evaluate_new_candidates
from .triangulate_candidates import triangulate_points
from utils.utility_tools import calculate_avarage_depth
from bootstrap.initialization import reinitialize
from structures import State

def process_frame(state: State, reinitialize_image, database_image, query_image, K, states_history):
  triangulate_signal = False
  min_number_keypoints = pl.params["min_number_keypoints"]
  max_inlier_ratio = pl.params["max_inlier_ratio"]

  # 4.1 we use KLT to track keypoints
  keypoints, landmarks = keypoint_association(state, database_image, query_image, K)

  # 4.2 we estimate the pose using PnP and recover R and t matrices
  R, t, inlier_keypoints, inlier_landmarks  = estimate_pose(state, landmarks, keypoints, K)

  state.update_state(inlier_keypoints.T, inlier_landmarks.T)

  avg_depth = calculate_avarage_depth(R, t, state.get_landmarks())

  re_initialize = keypoints.shape[0] < min_number_keypoints

  if re_initialize:
    M = states_history[-3]
    prev_R = M[:3,:3]
    prev_T = M[:, -1]
    state = reinitialize(reinitialize_image, query_image, K, prev_R, prev_T, R, t)


  # 4.3 we add new keypoints
  new_candidates_point = evaluate_new_candidates(state.get_all_keypoints(), query_image)

  # update the state of candidates point
  state.update_candidates_points(new_candidates_point.T)
  state.update_first_obs_candidates(new_candidates_point.T)
  Mvec = np.hstack((R, t[:,None])).flatten()
  N = new_candidates_point.shape[0]

  state.update_camera_pose_candidates(np.tile(Mvec, (N,1)).T)

  if not re_initialize:
    # triangulate new points from candidates
    triangulate_points(state, R, t, K, triangulate_signal)
  
  relative_position = -np.matmul(R.T, t)
  # absolute_position = np.matmul(prev_R.T, relative_position) + prev_t

  states_history.append(np.array(np.hstack((R, t[:,None]))))
  return relative_position, state, states_history
  
