"""
src.homography — Homography Estimation with RANSAC.

Computes the 3×3 projective transformation (homography) that maps
points from one image into the coordinate frame of another.
"""

import numpy as np
import cv2

from src.config import MIN_MATCH_COUNT, RANSAC_THRESH


def compute_homography(kp1, kp2, good_matches):
    """Estimate the 3x3 homography H (image-1 -> image-2) via DLT + RANSAC.

    Returns (H, inlier_mask) or (None, None) on failure.
    """
    if len(good_matches) < MIN_MATCH_COUNT:
        print(f"  [WARN] Only {len(good_matches)} matches "
              f"— need {MIN_MATCH_COUNT}.")
        return None, None

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts,
                                 cv2.RANSAC, RANSAC_THRESH)

    if H is None:
        print("  [WARN] Homography computation failed.")
        return None, None

    inliers = int(mask.sum())
    total   = len(good_matches)
    print(f"  Homography inliers: {inliers}/{total} "
          f"({100 * inliers / total:.1f}%)")
    return H, mask
