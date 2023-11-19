import cv2
import numpy as np
import matplotlib.pyplot as plt
from structures import State
from visualization_tools.point_cloud import plot_point_cloud, plot_feature_2D
from visualization_tools.plot_matches import plot_matched_points, plot_matched_points_with_lines

def initialization(frame1, frame2, K) -> State:
    # SIFT tunable parameters
    match_per_descriptor = 2
    match_treshold = 0.8
    # RANSAC tunable parameters
    repojection_error_tollerance = 0.9 # in pixels
    confidence = 0.9999

    # SIFT feature detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both frames
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)

    # Use a brute-force matcher to find matches between descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=match_per_descriptor)

    # Apply ratio test to get good matches
    good_matches = [m for m, n in matches if m.distance < match_treshold * n.distance]

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Find fundamental matrix using RANSAC
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.FM_RANSAC, repojection_error_tollerance, confidence)

    plot_matched_points_with_lines(frame1, src_pts, dst_pts, mask)
    
    # Filter out outliers
    src_pts = src_pts[mask.ravel() == 1]
    dst_pts = dst_pts[mask.ravel() == 1]

    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K)

    M1 = K @ np.eye(3, 4)
    M2 = K @ np.hstack((R, t))
 
    # Triangulate 3D points from the matches
    points_3d_homogeneous = cv2.triangulatePoints(M1, M2, src_pts.T, dst_pts.T)
   
    # Normalize homogeneous coordinates
    points_3d = points_3d_homogeneous / points_3d_homogeneous[-1,:]
 
    plot_point_cloud(points_3d, np.eye(3), t.flatten())
    plot_feature_2D(points_3d, t.flatten())

    return State(dst_pts, points_3d)