"""
src.homography — Homography Estimation with RANSAC.

Computes the 3×3 projective transformation (homography) that maps
points from one image into the coordinate frame of another.
"""

import numpy as np
import cv2

from src.config import MIN_MATCH_COUNT, RANSAC_THRESH


def compute_homography(kp1, kp2, good_matches):
    """Estimate the 3x3 homography H (image-1 -> image-2) via iterative RANSAC refinement.

    Uses two passes:
      1. USAC_MAGSAC with the configured threshold to find initial inliers.
      2. Re-estimate using only inliers with a tighter threshold for sub-pixel accuracy.

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

    # Pass 1: robust estimation
    try:
        H, mask = cv2.findHomography(src_pts, dst_pts,
                                     cv2.USAC_MAGSAC, RANSAC_THRESH,
                                     maxIters=10000, confidence=0.9999)
    except cv2.error:
        H, mask = cv2.findHomography(src_pts, dst_pts,
                                     cv2.RANSAC, RANSAC_THRESH,
                                     maxIters=10000, confidence=0.9999)

    if H is None:
        print("  [WARN] Homography computation failed.")
        return None, None

    inliers_pass1 = int(mask.sum())

    # Pass 2: re-estimate on inliers only with tighter threshold
    inlier_idx = mask.ravel().astype(bool)
    src_inliers = src_pts[inlier_idx]
    dst_inliers = dst_pts[inlier_idx]

    if len(src_inliers) >= MIN_MATCH_COUNT:
        try:
            H2, mask2 = cv2.findHomography(src_inliers, dst_inliers,
                                           cv2.USAC_MAGSAC, RANSAC_THRESH * 0.5,
                                           maxIters=10000, confidence=0.9999)
        except cv2.error:
            H2, mask2 = cv2.findHomography(src_inliers, dst_inliers,
                                           cv2.RANSAC, RANSAC_THRESH * 0.5)

        if H2 is not None:
            inliers_pass2 = int(mask2.sum())
            # Only use refined result if it kept most inliers
            if inliers_pass2 >= inliers_pass1 * 0.5:
                H = H2
                # Pass 3: final polish on tightest inliers
                inlier_idx2 = mask2.ravel().astype(bool)
                src_final = src_inliers[inlier_idx2]
                dst_final = dst_inliers[inlier_idx2]
                if len(src_final) >= MIN_MATCH_COUNT:
                    try:
                        H3, mask3 = cv2.findHomography(src_final, dst_final,
                                                       cv2.LMEDS)
                        if H3 is not None:
                            H = H3
                    except cv2.error:
                        pass

    total = len(good_matches)
    print(f"  Homography inliers: {inliers_pass1}/{total} "
          f"({100 * inliers_pass1 / total:.1f}%)")
    return H, mask
