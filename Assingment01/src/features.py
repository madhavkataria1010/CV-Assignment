"""
src.features — SIFT Feature Extraction & Matching.

This module handles:
  1. Detecting SIFT keypoints and computing 128-D descriptors.
  2. Matching descriptors between image pairs using FLANN + Lowe's ratio test.
"""

import os
import cv2
import numpy as np

from src.config import SIFT_NFEATURES, LOWE_RATIO


def extract_sift_features(images: list, output_dir: str = None):
    """
    Detect SIFT keypoints and compute 128-D descriptors for every image.

    SIFT (Scale-Invariant Feature Transform) — Lowe, 2004 — works by:
      1. Building a scale-space via Difference-of-Gaussian (DoG) pyramids.
      2. Localising extrema in (x, y, scale) space → keypoint candidates.
      3. Assigning dominant orientation(s) per keypoint.
      4. Computing a 4×4 grid of 8-bin orientation histograms → 128-D
         descriptor vector, normalised for illumination invariance.

    Parameters
    ----------
    images     : list[np.ndarray]   — BGR images
    output_dir : str or None        — if given, save keypoint visualisations

    Returns
    -------
    keypoints   : list[tuple[cv2.KeyPoint, ...]]
    descriptors : list[np.ndarray]  — each shape (N, 128), dtype float32
    """
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    keypoints_list, descriptors_list = [], []

    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
        print(f"  Image {i}: {len(kp)} keypoints detected")

        if output_dir:
            vis = cv2.drawKeypoints(
                img, kp, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            cv2.imwrite(os.path.join(output_dir, f"keypoints_{i}.jpg"), vis)

    print()
    return keypoints_list, descriptors_list


def match_features(des1, des2, kp1, kp2,
                   img1=None, img2=None,
                   output_dir=None, pair_label=""):
    """
    Match SIFT descriptors between two images.

    Uses FLANN (Fast Library for Approximate Nearest Neighbours) with a
    KD-tree index for efficient k-NN queries, followed by **Lowe's ratio
    test**: a match ``m`` is kept only when its distance to the best
    neighbour is significantly smaller than to the second-best ``n``:

        keep  iff  m.distance < ratio × n.distance

    This eliminates ~90 % of false matches while retaining ~95 % of true
    ones (Lowe, 2004).

    Parameters
    ----------
    des1, des2     : np.ndarray  — descriptor arrays
    kp1, kp2       : keypoint tuples
    img1, img2     : original images (optional, for visualisation)
    output_dir     : save visualisation here
    pair_label     : label for the image pair

    Returns
    -------
    good_matches : list[cv2.DMatch]
    """
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < LOWE_RATIO * n.distance:
                good.append(m)

    print(f"  {pair_label}: {len(good)} good matches "
          f"(from {len(matches)} raw)")

    if output_dir and img1 is not None and img2 is not None:
        vis = cv2.drawMatches(
            img1, kp1, img2, kp2, good, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(
            os.path.join(output_dir, f"matches_{pair_label}.jpg"), vis,
        )

    return good
