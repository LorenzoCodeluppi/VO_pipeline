import cv2
import numpy as np

from structures import State

def estimate_pose(state: State, tracked_landmarks, tracked_keypoints, K):
  assert tracked_landmarks.shape[1] == 3, "Landmarks should be in 3D (N, 3)"
  assert tracked_keypoints.shape[1] == 2, "Keypoints should be in 2D (N, 2)"

  # PnP parameters
  reprojection_error = 1
  confidence = 0.9999

  # Use solvePnPRansac to estimate pose (R, t)
  _, rvec, tvec, inliers = cv2.solvePnPRansac(
    tracked_landmarks,
    tracked_keypoints,
    K,
    None,
    reprojectionError= reprojection_error,
    confidence= confidence,
  )
  t = tvec.flatten()

  R, _ = cv2.Rodrigues(rvec)

  # Use inliers for further processing if needed
  inlier_keypoints = tracked_keypoints[inliers.ravel()]
  inlier_landmarks = tracked_landmarks[inliers.ravel()]

  return R, -np.matmul(R.T, t), inlier_keypoints, inlier_landmarks