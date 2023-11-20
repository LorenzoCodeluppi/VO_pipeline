import cv2
import numpy as np

from structures import State

# KLT has no bounds, point can be outside of the image... need a smart fix
def keypoint_association(state: State, database_image, query_image, K):
  previous_keypoints = state.get_keypoints()
  landmarks = state.get_landmarks()
  candidates_points = state.get_candidates_points()
  image_height, image_width = query_image.shape

  lk_params = dict(winSize=(10, 10),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

  next_keypoints, status, err = cv2.calcOpticalFlowPyrLK(
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
   
    state.filter_out_candidates(next_candidates_keypoints.T, candidates_status.flatten())

  if status is None:
    return
  
  tracked_keypoints = next_keypoints[status.ravel() == 1]
  tracked_landmarks = landmarks[:, status.ravel() == 1]

  return tracked_keypoints, tracked_landmarks.T
  
  