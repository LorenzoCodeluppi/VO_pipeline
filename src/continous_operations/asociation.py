import cv2
import numpy as np

import params_loader as pl
from utils.utility_tools import get_validation_mask
#from klt.track_klt_robustly import trackKLTRobustly
from structures import State

# KLT has no bounds, point can be outside of the image... need a smart fix
def keypoint_association(state: State, database_image, query_image, K):
    previous_keypoints = state.get_keypoints()
    landmarks = state.get_landmarks()
    candidates_points = state.get_candidates_points()

    error_threshold = pl.params["error_threshold"]
    win_size = pl.params["winSize"]
    max_level = pl.params["maxLevel"]
    criteria = pl.params["criteria"]


    lk_params = dict(winSize=win_size,
        maxLevel=max_level,
        criteria=criteria)


    next_keypoints, keypoints_status, keypoints_err = cv2.calcOpticalFlowPyrLK(database_image, query_image, previous_keypoints.T, None, **lk_params)

    if candidates_points is not None:
        next_candidates_keypoints, candidates_status, candidates_err = cv2.calcOpticalFlowPyrLK(database_image, query_image, candidates_points.T.astype(np.float32), None, **lk_params)
        candidates_mask = get_validation_mask(candidates_status, candidates_err, error_threshold)

        if next_candidates_keypoints is not None:
            state.filter_out_candidates(next_candidates_keypoints.T, candidates_mask.flatten())

    if keypoints_status is None:
        return

    keypoints_mask = get_validation_mask(keypoints_status, keypoints_err, error_threshold)
    tracked_keypoints = next_keypoints[keypoints_mask.flatten() == 1]
    tracked_landmarks = landmarks[:, keypoints_mask.flatten() == 1]

    return tracked_keypoints, tracked_landmarks.T

