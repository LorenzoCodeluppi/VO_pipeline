import cv2
import numpy as np

from structures import State

def triangulate_points(state: State, current_R, current_t, K):
  candidates = state.get_candidates_points()
  first_obs_candidates = state.get_first_obs_candidates()
  poses = state.get_camera_pose_candidates().reshape(3,4, candidates.shape[1])
  T = poses[:,-1,:]

  # parameters to tune
  distance_threshold = 5

  # calculate the distance between each poses to the current pose (t), if > than threshold select them
  distances = np.linalg.norm(T - current_t[:,None], axis=0)
  mask = distances > distance_threshold

  possible_new_landmarks = np.sum(mask)

  if possible_new_landmarks > 0:
    prev_poses = poses[:,:,mask]

    # now we assume that all the poses are the same, but this is not true in general, need a fix
    prev_pose = prev_poses[:,:,0]

    # Chain transformations to get the poses in respect to the previous pose
    # relative_R = np.linalg.inv(prev_pose[:,:3]) @ current_R  # R_relative = R_prev^-1 * R_current
    # relative_t = np.linalg.inv(prev_pose[:,:3]) @ (current_t - prev_pose[:,-1])  # t_relative = R_prev^-1 * (t_current - t_prev)

    current_pose = np.hstack((current_R, current_t[:,None]))

    M1 = K @ prev_pose
    M2 = K @ current_pose

    selected_candidates = candidates[:, mask].astype(np.float32)
    selected_first_obs_candidates = first_obs_candidates[:, mask].astype(np.float32)
 
    points_3d_homogeneous = cv2.triangulatePoints(M1, M2, selected_first_obs_candidates, selected_candidates)

    points_3d = points_3d_homogeneous[:3,:]
    # print(points_3d)
    state.move_candidates_to_keypoints(selected_candidates, points_3d, ~mask)
