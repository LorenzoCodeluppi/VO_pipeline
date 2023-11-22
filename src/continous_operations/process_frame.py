import numpy as np

from .asociation import keypoint_association
from .estimate_pose import estimate_pose
from .evaluate_points import evaluate_new_candidates
from .triangulate_candidates import triangulate_points
from structures import State

def process_frame(state: State, database_image, query_image, K):
  # 4.1 we use KLT to track keypoints
  keypoints, landmarks = keypoint_association(state, database_image, query_image, K)

  # 4.2 we estimate the pose using PnP and recover R and t matrices
  R, t, inlier_keypoints, inlier_landmarks  = estimate_pose(state, landmarks, keypoints, K)
  state.update_state(inlier_keypoints.T, inlier_landmarks.T)

  # 4.3 we add new keypoints
  new_candidates_point = evaluate_new_candidates(state.get_all_keypoints(), query_image)

  # update the state of candidates point
  state.update_candidates_points(new_candidates_point.T)
  state.update_first_obs_candidates(new_candidates_point.T)
  Mvec = np.hstack((R, t[:,None])).flatten()
  N = new_candidates_point.shape[0]

  state.update_camera_pose_candidates(np.tile(Mvec, (N,1)).T)

  # triangulate new points from candidates
  triangulate_points(state, R, t, K)

  return -np.matmul(R.T, t)
  