import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

from .draw_camera import drawCamera

def calculate_axis_limits(points_3d):
    """
    Calculate axis limits for a 3D point cloud.

    Parameters:
    - points_3d: numpy array of shape (3, N) representing the 3D points.

    Returns:
    - axis_limits: List of tuples, each containing (min_value, max_value) for each axis.
    """
    min_values = np.min(points_3d, axis=1)
    max_values = np.max(points_3d, axis=1)

    axis_limits = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

    return axis_limits

def plot_point_cloud(points_3d, R, t, set_axis_limit=True, equal_axis=False):
  # Extract x, y, and z coordinates from the 3D points
  x = points_3d[0, :]
  y = points_3d[1, :]
  z = points_3d[2, :]

  # Create a 3D scatter plot
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(x, y, z, marker='o')
  drawCamera(ax, t, R, length_scale=10)

  if not set_axis_limit:
    return

  axis_limit = calculate_axis_limits(points_3d)
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  zlim = ax.get_zlim()

  ax.set_xlim([min(xlim[0], axis_limit[0][0]), max(xlim[1], axis_limit[0][1])])
  ax.set_ylim([min(ylim[0], axis_limit[1][0]), max(ylim[1], axis_limit[1][1])])
  ax.set_zlim([min(zlim[0], axis_limit[2][0]), max(zlim[1], axis_limit[2][1])])
  
  # This sets the aspect ratio to 'equal'
  if equal_axis:
    ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))

  plt.show()


def plot_feature_2D(points_3d, t, set_axis_limit=True, equal_axis=False):
  # Extract x, y, and z coordinates from the 3D points
  x = points_3d[0, :]
  z = points_3d[2, :]

  # Create a 3D scatter plot
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.scatter(x, z, marker='o')
  ax.scatter(t[0], t[2], marker='x')

  if not set_axis_limit:
    return

  axis_limit = calculate_axis_limits(points_3d)
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()

  ax.set_xlim([min(xlim[0], axis_limit[0][0]), max(xlim[1], axis_limit[0][1])])
  ax.set_ylim([min(ylim[0], axis_limit[2][0]), max(ylim[1], axis_limit[2][1])])
  
  # This sets the aspect ratio to 'equal'
  if equal_axis:
    ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))

  plt.show()