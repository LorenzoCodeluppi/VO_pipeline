import cv2
import numpy as np

from structures import State
from utils import utility_tools as utils

def triangulate_points(state: State, current_R, current_t, K):
  candidates = state.get_candidates_points()
  first_obs_candidates = state.get_first_obs_candidates()
  poses = state.get_camera_pose_candidates().reshape(3,4, candidates.shape[1])
  T = poses[:,-1,:] # 3xN
  
# TRIAL WITH BEARING VECTORS FAIL
  current_camera_pose_vec = np.tile((current_t[(1,2),None]), (1, candidates.shape[1]))
  #Compute bearing vectors for current candidate point and current camera pose 
  bearing_vector = utils.bearingvector(candidates,current_camera_pose_vec)
  #Compute bearing vectors for first observation and first camera pose
  first_obs_bearing_vector = utils.bearingvector(first_obs_candidates, T[(1,2),:])

  #Compute the angle between the two bearing vectors
  angle_between = utils.angle(bearing_vector, first_obs_bearing_vector)


  # parameters to tune
  distance_threshold = 5

  # calculate the distance between each poses to the current pose (t), if > than threshold select them
  distances = np.linalg.norm(T - current_t[:,None], axis=0)

  mask = distances > distance_threshold

  possible_new_landmarks = np.sum(mask)
  

  if possible_new_landmarks > 0:
    prev_poses = poses[:,:,mask]

    unique, indices, counts = np.unique(prev_poses, return_index = True, return_counts= True, axis = 2)
    # print(indices)
    # now we assume that all the poses are the same, but this is not true in general, need a fix
    prev_pose = prev_poses[:,:,0]
    # print(prev_pose)

    current_pose = np.hstack((current_R, current_t[:,None]))

    M1 = K @ prev_pose
    M2 = K @ current_pose

    selected_candidates = candidates[:, mask].astype(np.float32)
    selected_first_obs_candidates = first_obs_candidates[:, mask].astype(np.float32)
 
    points_3d_homogeneous = cv2.triangulatePoints(M1, M2, selected_first_obs_candidates, selected_candidates)

    points_3d = points_3d_homogeneous[:3,:] / points_3d_homogeneous[-1, :]
    # print(state.get_landmarks().shape)

    state.move_candidates_to_keypoints(selected_candidates, points_3d, ~mask)
    # print(state.get_landmarks().shape)

