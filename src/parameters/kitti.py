import cv2

kitti_params_map = {
    "bootstrap_frames": [0, 3],

    # bootstrap parameters (initialization)
    "match_per_descriptor": 2, # SIFT 
    "match_treshold": 0.8,     # SIFT
    "repojection_error_tollerance": 0.9, # Essential matrix Ransac (in pixels)
    "p5p_confidence": 0.9999, # Essential matrix Ransac

    # KLT (association)
    "error_threshold": 10, # KLT
    "winSize": (30, 30), # KLT
    "maxLevel": 3, # KLT
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001), # KLT

    # PnP (estimate_pose)
    "reprojection_error": 1, # PnP RANSAC
    "pnp_confidence_" : 0.999, # PnP RANSAC

    # Harris (evaluate_points)
    "block_size" : 7, # Harris block_size
    "k_size" : 5, # Harris k_zie
    "k" : 0.04, # Harris magic number
    "num_corners" : 300, # Harris num of corner to extract
    "suppression_radius" : 15, # Harris suppression radius
    "min_distance" : 8, # Harris minimum distance from other keypoints

    # Triangulate (triangulate_candidates, process_frame)
    "distance_threshold": 1, # Distance to the previous frames in start triangulate
    "min_number_keypoints": 120 ,# Minimum number of keypoints to have
    "max_inlier_ratio": 0.3, # Keypoints cannot drop under this ratio between 2 consecutives frames
    "thumb_rule": 0.2, # Rule to start trinagulate points
    "min_reprojection_error": 1,
    "distance_threshold_factor": 2,

}
