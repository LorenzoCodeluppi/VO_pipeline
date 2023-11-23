import cv2
import numpy as np

from structures import State

# KLT has no bounds, point can be outside of the image... need a smart fix
def keypoint_association(state: State, database_image, query_image, K):
  previous_keypoints = state.get_keypoints()
  landmarks = state.get_landmarks()
  candidates_points = state.get_candidates_points()
  image_height, image_width = query_image.shape
  error_threshold = 5

  lk_params = dict(winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

  next_keypoints, status, err = cv2.calcOpticalFlowPyrLK(
    database_image,
    query_image,
    previous_keypoints.T,
    None,
    **lk_params
  )
  # print(err)
  if candidates_points is not None:
    next_candidates_keypoints, candidates_status, candidates_err = cv2.calcOpticalFlowPyrLK(
      database_image,
      query_image,
      candidates_points.T.astype(np.float32),
      None,
      **lk_params
    )

    valid_candidates_mask = candidates_err < error_threshold
    # print(f"Number of valid candidates: {next_candidates_keypoints.shape}")
    # next_candidates_keypoints = next_candidates_keypoints[valid_candidates_mask.flatten() ==1]

    mask_de_cristo = np.logical_and(valid_candidates_mask,candidates_status)
    print(f"Number of maske valid candidate: {np.sum(mask_de_cristo)}")
    if next_candidates_keypoints is not None:
      state.filter_out_candidates(next_candidates_keypoints.T, mask_de_cristo.flatten())

    print(f"Number of candidates: {candidates_err.shape}")

  if status is None:
    return
  
  
  tracked_keypoints = next_keypoints[status.ravel() == 1]
  tracked_landmarks = landmarks[:, status.ravel() == 1]

  return tracked_keypoints, tracked_landmarks.T
  
  