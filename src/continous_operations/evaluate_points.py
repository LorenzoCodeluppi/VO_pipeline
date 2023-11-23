import cv2
import numpy as np
import matplotlib.pyplot as plt

from structures import State

def evaluate_new_candidates(tracked_keypoints, query_image):

  # Harris parameters
  block_size = 4
  k_size = 5
  k = 0.04
  threshold = 0.04  #Best performance so far 
  min_distance = 10 # not sure about this

  corners = cv2.cornerHarris(query_image, blockSize=block_size, ksize=k_size, k=k)
  _, corners_binary = cv2.threshold(corners, threshold * corners.max(), 255, 0)

  new_keypoints = np.transpose(np.nonzero(corners_binary.T))

  # TODO: check
  new_keypoints = np.array(new_keypoints)
  distances = np.linalg.norm(new_keypoints[:, np.newaxis, :] - tracked_keypoints.T, axis=2)
  is_close = np.any(distances < min_distance, axis=1)
  new_keypoints_filtered = new_keypoints[~is_close]
  
  return new_keypoints_filtered
