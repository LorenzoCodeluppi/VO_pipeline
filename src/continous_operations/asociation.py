import cv2
import numpy as np

from utils.utility_tools import get_validation_mask
from structures import State

# KLT has no bounds, point can be outside of the image... need a smart fix
def keypoint_association(state: State, database_image, query_image, K):
  previous_keypoints = state.get_keypoints()
  landmarks = state.get_landmarks()
  candidates_points = state.get_candidates_points()
  error_threshold = 10

  lk_params = dict(winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

  next_keypoints, keypoints_status, keypoints_err = cv2.calcOpticalFlowPyrLK(
    database_image,
    query_image,
    previous_keypoints.T,
    None,
    **lk_params
  )

  if candidates_points is not None:
    next_candidates_keypoints, candidates_status, candidates_err = cv2.calcOpticalFlowPyrLK(
      database_image,
      query_image,
      candidates_points.T.astype(np.float32),
      None,
      **lk_params
    )
    candidates_mask = get_validation_mask(candidates_status, candidates_err, error_threshold)
  
    if next_candidates_keypoints is not None:
      state.filter_out_candidates(next_candidates_keypoints.T, candidates_mask.flatten())

  if keypoints_status is None:
    return
  
  keypoints_mask = get_validation_mask(keypoints_status, keypoints_err, error_threshold)
  tracked_keypoints = next_keypoints[keypoints_mask.flatten() == 1]
  tracked_landmarks = landmarks[:, keypoints_mask.flatten() == 1]

  return tracked_keypoints, tracked_landmarks.T
  
  