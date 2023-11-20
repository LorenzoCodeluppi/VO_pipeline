import cv2
import numpy as np
import matplotlib.pyplot as plt

from structures import State

def get_candidate_points(tracked_keypoints, query_image):

  # Harris parameters
  block_size = 2
  k_size = 3
  k = 0.04
  threshold = 0.01
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


def triangulate_points(state: State, query_image):
  candidate_keypoints = get_candidate_points(state.get_all_keypoints(), query_image)
  return candidate_keypoints