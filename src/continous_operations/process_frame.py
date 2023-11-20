import numpy as np

from .asociation import keypoint_association
from .estimate_pose import estimate_pose
from .triangulate_points import triangulate_points
from structures import State

def process_frame(previous_state: State, database_image, query_image, K):
  # 4.1 we use KLT to track keypoints
  keypoints, landmarks = keypoint_association(previous_state, database_image, query_image, K)

  # 4.2 we estimate the pose using PnP and recover R and t matrices
  R, t,inlier_keypoints, inlier_landmarks  = estimate_pose(previous_state, landmarks, keypoints, K)
  previous_state.update_state(inlier_keypoints.T, inlier_landmarks.T)

  # 4.3 we add new keypoints
  new_candidates_point = triangulate_points(previous_state, query_image)

  previous_state.update_candidates_points(new_candidates_point.T)
  previous_state.update_first_obs_candidates(new_candidates_point.T)
  Mvec = np.hstack((R, t[:,None])).flatten()
  N = new_candidates_point.shape[0]
  previous_state.update_camera_pose_candidates(np.tile(Mvec, N).reshape(12, N))

  return t
  