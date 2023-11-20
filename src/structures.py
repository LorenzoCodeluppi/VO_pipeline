from enum import Enum

class Dataset(Enum):
  KITTI = 0
  MALAGA = 1
  PARKING = 2

class State():
  keypoints = None
  landmarks = None

  # only INLIERS
  def __init__(self, keypoints, landmarks):
    self.keypoints = keypoints
    self.landmarks = landmarks

  def update_state(self, new_keypoints, new_landmarks):
    self.keypoints = new_keypoints
    self.landmarks = new_landmarks

  def get_keypoints(self):
    return self.keypoints

  def get_landmarks(self):
    return self.landmarks