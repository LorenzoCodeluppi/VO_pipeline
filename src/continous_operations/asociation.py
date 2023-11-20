import cv2
import numpy as np

from structures import State

def keypoint_association(state: State, database_image, query_image, K):
  previous_keypoints = state.get_keypoints()
  landmarks = state.get_landmarks()
  
  lk_params = dict(winSize=(10, 10),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.0003))

  next_keypoints, status, err = cv2.calcOpticalFlowPyrLK(
    database_image,
    query_image,
    previous_keypoints,
    None,
    **lk_params
  )
  if status is None:
    return
  
  tracked_keypoints = next_keypoints[status.ravel() == 1]
  tracked_landmarks = landmarks[status.ravel() == 1]

  return tracked_keypoints, tracked_landmarks
  
  