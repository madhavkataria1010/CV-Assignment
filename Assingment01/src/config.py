"""
src.config — Central configuration constants.

All tuneable parameters live here so they can be imported by any module
without circular dependencies.
"""

SIFT_NFEATURES  = 10000     # Max SIFT keypoints per image
MIN_MATCH_COUNT = 10        # Minimum good matches to attempt homography
LOWE_RATIO      = 0.70      # Lowe's ratio-test threshold
RANSAC_THRESH   = 1.5       # RANSAC reprojection error threshold (pixels)
PYRAMID_LEVELS  = 6         # Laplacian pyramid depth for multi-band blend
MAX_DIMENSION   = 1200      # Resize longest side to this (0 = no resize)
