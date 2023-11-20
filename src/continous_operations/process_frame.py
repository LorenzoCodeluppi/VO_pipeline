from .asociation import keypoint_association
from .estimate_pose import estimate_pose

def process_frame(previous_state, database_image, query_image, K):
  keypoints, landmarks = keypoint_association(previous_state, database_image, query_image, K)
  t = estimate_pose(previous_state, landmarks, keypoints, K)
  return t
  