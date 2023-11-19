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