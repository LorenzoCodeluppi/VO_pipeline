import numpy as np
import matplotlib.pyplot as plt

def plot_matched_points(image, keypoints, inlier_mask, plot_outliers=False):
    """
    Plot the matched points on the image.

    Parameters:
    - image: The image.
    - keypoints: 2D array of shape (N, 2) representing the matched keypoints.
    - inlier_mask: Boolean array of shape (N,) indicating whether each match is an inlier.

    Returns:
    - None
    """
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the image
    ax.imshow(image, cmap='gray')

    # Plot the matched keypoints
    ax.plot(keypoints[inlier_mask.ravel() == 1, 0],
            keypoints[inlier_mask.ravel() == 1, 1],
            'rx', markersize=4)  # Mark inlier matches in red
    if plot_outliers:
      ax.plot(keypoints[inlier_mask.ravel() == 0, 0],
              keypoints[inlier_mask.ravel() == 0, 1],
              'bx', markersize=4)  # Mark outlier matches in blue
      
    ax.set_title('Matched Points on Image')
    plt.show()


def plot_matched_points_with_lines(database_image, database_keypoints, query_keypoints, inlier_mask):
    """
    Plot the matched points from both query and database and also plot the line connecting them.

    Parameters:
    - database_image: The database image.
    - query_image: The query image.
    - database_keypoints: 2D array of shape (N, 2) representing the matched keypoints in the database image.
    - query_keypoints: 2D array of shape (N, 2) representing the matched keypoints in the query image.
    - inlier_mask: Boolean array of shape (N,) indicating whether each match is an inlier.

    Returns:
    - None
    """
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the database image
    ax.imshow(database_image, cmap='gray')

    # Plot the matched keypoints
    ax.plot(database_keypoints[inlier_mask.ravel() == 1, 0],
            database_keypoints[inlier_mask.ravel() == 1, 1],
            'rx', markersize=2)
    
    ax.plot(query_keypoints[inlier_mask.ravel() == 1, 0],
            query_keypoints[inlier_mask.ravel() == 1, 1],
            'rx', markersize=2) 

    # Plot lines connecting matched points
    for i in range(len(inlier_mask)):
        if inlier_mask[i]:
            ax.plot([database_keypoints[i, 0], query_keypoints[i, 0]],
                    [database_keypoints[i, 1], query_keypoints[i, 1]],
                    'g-', linewidth=1)  # Connect matched points with green lines

    ax.set_title('Matchings')

    plt.show()