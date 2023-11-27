import cv2
import numpy as np
import matplotlib.pyplot as plt

import params_loader as pl
from structures import State

def selectKeypoints(scores, num, r):
    """
    Selects the num best scores as keypoints and performs non-maximum suppression of a (2r + 1)*(2r + 1) box around
    the current maximum.
    """

    keypoints = np.zeros([num, 2])
    temp_scores = np.pad(scores, [(r, r), (r, r)], mode='constant', constant_values=0)

    for i in range(num):
        kp = np.unravel_index(temp_scores.argmax(), temp_scores.shape)
        keypoints[i, :] = np.array(kp)[::-1] - r  # Swap row and column, then subtract r
        temp_scores[(kp[0] - r):(kp[0] + r + 1), (kp[1] - r):(kp[1] + r + 1)] = 0

    return keypoints

def evaluate_new_candidates(tracked_keypoints, query_image):

  # Harris parameters
  block_size = pl.params["block_size"]
  k_size = pl.params["k_size"]
  k = pl.params["k"]
  num_corners = pl.params["num_corners"]
  suppression_radius = pl.params["suppression_radius"]
  min_distance = pl.params["min_distance"] # not sure about this

  corners = cv2.cornerHarris(query_image, blockSize=block_size, ksize=k_size, k=k)
  # _, corners_binary = cv2.threshold(corners, 0.2 * corners.max(), 255, 0)

  keypoints = selectKeypoints(corners, num_corners, suppression_radius)

  # TODO: check
  distances = np.linalg.norm(keypoints[:,None,:] - tracked_keypoints.T, axis=2)
  is_close = np.any(distances < min_distance, axis=1)
  new_keypoints_filtered = keypoints[~is_close]

  return new_keypoints_filtered
