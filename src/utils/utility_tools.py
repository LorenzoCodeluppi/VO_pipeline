import numpy as np
from pathlib import Path
import os 

def cross2Matrix(x):
  """ Antisymmetric matrix corresponding to a 3-vector
    Computes the antisymmetric matrix M corresponding to a 3-vector x such
    that M*y = cross(x,y) for all 3-vectors y.

    Input: 
      - x np.ndarray(3,1) : vector

    Output: 
      - M np.ndarray(3,3) : antisymmetric matrix
  """
  M = np.array([[0,   -x[2], x[1]], 
                [x[2],  0,  -x[0]],
                [-x[1], x[0],  0]])
  return M

def bearingvector(point, pose):
  """ Bearing vector from camera pose to 2d point in camera frame
    Computes the angle between two camera-image plane vector

    Input:
     - point np.ndarray(3,1) : vector
     - pose np.ndattay(3,1) : vector

    Output:

     - bearing_vector np.ndarray(3,1) : vector
   """
  
  v = point - pose / np.abs(point - pose)
  return v



def angle(v1,v2):
  """ Angle between two vectors
    Computes the angle between two vectors.

    Input: 
      - v1 np.ndarray(3,1) : first vector
      - v2 np.ndarray(3,1) : second vector

    Output: 
      - angle float : angle between the two vectors
  """
  v1 = v1 / np.linalg.norm(v1)
  v2 = v2 / np.linalg.norm(v2)
  angle = np.arccos(np.dot(v1.T,v2))
  angle = np.rad2deg(angle)
  return angle

def calculate_inlier_ratio(previous_keypoints, inlier_keypoints):
  number_current_keypoints = inlier_keypoints.shape[0]
  number_previous_keypoints = previous_keypoints.shape[1]
  keypoints_ratio = number_current_keypoints / number_previous_keypoints
  
  return keypoints_ratio

def calculate_avarage_depth(landmarks, R, t):
  position = -np.matmul(R.T, t)
  return np.mean(landmarks[-1,:]) - position[-1]

def get_validation_mask(status, error, threshold):
  if error is not None:
    valid_keypoints_mask = error < threshold
    return np.logical_and(valid_keypoints_mask, status)
  return status

def load_Kitti_GT():

  ROOT_DIR = Path(__file__).parent.parent.parent  

  data_folder_path = str(ROOT_DIR) + '/data/kitti/poses'

  # Initialize an empty array to store poses
  all_poses = []

# Loop through each .txt file in the directory
  for file_name in sorted(os.listdir(data_folder_path)):
      if file_name.endswith(".txt"):
          file_path = os.path.join(data_folder_path, file_name)
        
          # Load poses from the current file
          poses = np.loadtxt(file_path)
          
          # Append the poses to the array
          all_poses.append(poses)

  # Concatenate all loaded poses into a single array
  all_poses_array = np.concatenate(all_poses, axis=0)
  
  # Extract the first three columns of the array
  # all_poses_array = all_poses_array[:, :3]
  
 
  return  all_poses_array
