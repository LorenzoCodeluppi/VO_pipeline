import numpy as np
from enum import Enum

class Dataset(Enum):
  KITTI = 0
  MALAGA = 1
  PARKING = 2

class State():
  keypoints = None #2XN
  landmarks = None #3XN
  candidates_points = None #2XM
  first_obs_candidates = None #2XM
  camera_pose_candidates = None #12XM

  # only INLIERS
  def __init__(self, keypoints, landmarks):
    self.keypoints = keypoints
    self.landmarks = landmarks

  def update_state(self, new_keypoints, new_landmarks, update = False):
    self.keypoints = new_keypoints
    self.landmarks = new_landmarks
    # if not update:
      # self.keypoints = new_keypoints
      # self.landmarks = new_landmarks
    # else:
    #   self.keypoints = np.concatenate((self.keypoints, new_keypoints), axis=1)
    #   self.landmarks = np.concatenate((self.landmarks, new_landmarks), axis=1)
  
  def update_candidates_points(self, new_candidates_points, replace = False):
    assert new_candidates_points.shape[0] == 2; "Wrong candidate points dimension"

    if self.candidates_points is None or replace is True:
      self.candidates_points = new_candidates_points
    else:
      self.candidates_points = np.concatenate((self.candidates_points, new_candidates_points), axis=1)

  def update_first_obs_candidates(self, new_first_obs_candidates, replace = False):
    assert new_first_obs_candidates.shape[0] == 2; "Wrong candidate points dimension"

    if self.first_obs_candidates is None or replace is True:
      self.first_obs_candidates = new_first_obs_candidates
    else:
      self.first_obs_candidates = np.concatenate((self.first_obs_candidates, new_first_obs_candidates), axis=1)

  # To be defined
  def update_camera_pose_candidates(self, new_camera_pose_candidates, replace = False):
    assert new_camera_pose_candidates.shape[0] == 12; "Wrong candidate points dimension"

    if self.camera_pose_candidates is None or replace is True:
      self.camera_pose_candidates = new_camera_pose_candidates
    else:
      self.camera_pose_candidates = np.concatenate((self.camera_pose_candidates, new_camera_pose_candidates), axis=-1)

  def filter_out_candidates(self, new_tracked_points, mask):
    if self.candidates_points is None:
      raise Exception
    
    self.update_candidates_points(new_tracked_points, replace=True)
    self.update_candidates_points(self.candidates_points[:,mask == 1], replace=True)
    self.update_first_obs_candidates(self.first_obs_candidates[:,mask == 1], replace=True)
    self.update_camera_pose_candidates(self.camera_pose_candidates[:,mask == 1], replace=True)

  def move_candidates_to_keypoints(self, new_keypoints, new_landmarks, filter_mask):
    self.update_state(new_keypoints, new_landmarks, update=True)
    self.filter_out_candidates(self.get_candidates_points(), filter_mask)
    
  def get_all_keypoints(self):
    if self.keypoints is None:
      raise Exception
    if self.candidates_points is None:
      return self.keypoints
    else:
      return np.concatenate((self.keypoints, self.candidates_points), axis=1)

  def get_keypoints(self):
    return self.keypoints

  def get_landmarks(self):
    return self.landmarks
  
  def get_candidates_points(self):
    return self.candidates_points
  
  def get_first_obs_candidates(self):
    return self.first_obs_candidates
  
  def get_camera_pose_candidates(self):
    return self.camera_pose_candidates
