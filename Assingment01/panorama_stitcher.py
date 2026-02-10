"""Panorama Stitching Pipeline -- CV Assignment 1.

Stitches 3+ overlapping images into a panorama using SIFT, homography
estimation, and multiple blending strategies.

Usage:
    python panorama_stitcher.py --images_dir images/ --method all
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

from src.config import PYRAMID_LEVELS, MAX_DIMENSION
from src.features import extract_sift_features, match_features
from src.homography import compute_homography
from src.warping import (
    compute_canvas_size, warp_image,
    cylindrical_warp_vectorized, trim_black_borders,
)
from src.blending import naive_stitch, linear_blend, multiband_blend_all
from src.utils import load_images, gain_compensate, crop_black, generate_comparison


def standard_pipeline(images, output_dir):
    """SIFT + FLANN + RANSAC homography + warp + blend (centre-referenced)."""
    print("=" * 60)
    print("  STANDARD PIPELINE")
    print("=" * 60)

    n = len(images)
    ref = n // 2
    print(f"  Reference image: {ref}\n")

    kps, descs = extract_sift_features(images, output_dir)

    homographies = [None] * n
    homographies[ref] = np.eye(3, dtype=np.float64)

    for i in range(ref - 1, -1, -1):
        good = match_features(descs[i], descs[i + 1], kps[i], kps[i + 1],
                              images[i], images[i + 1], output_dir,
                              pair_label=f"{i}-{i+1}")
        H, _ = compute_homography(kps[i], kps[i + 1], good)
        if H is None:
            print(f"  [ERROR] Cannot stitch image {i}.")
            return None, None, None
        homographies[i] = homographies[i + 1] @ H

    for i in range(ref + 1, n):
        good = match_features(descs[i], descs[i - 1], kps[i], kps[i - 1],
                              images[i], images[i - 1], output_dir,
                              pair_label=f"{i}-{i-1}")
        H, _ = compute_homography(kps[i], kps[i - 1], good)
        if H is None:
            print(f"  [ERROR] Cannot stitch image {i}.")
            return None, None, None
        homographies[i] = homographies[i - 1] @ H

    canvas_size, offset = compute_canvas_size(images, homographies)
    T = np.array([[1, 0, offset[0]],
                  [0, 1, offset[1]],
                  [0, 0, 1]], dtype=np.float64)

    warped_images = []
    for i, (img, H) in enumerate(zip(images, homographies)):
        w = warp_image(img, T @ H, canvas_size)
        warped_images.append(w)
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"warped_{i}.jpg"), w)

    naive = crop_black(naive_stitch(warped_images))
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "naive_stitch.jpg"), naive)

    warped_comp = gain_compensate(warped_images)

    blended_lin = crop_black(linear_blend(warped_comp))
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "panorama_linear.jpg"), blended_lin)

    blended_mb = crop_black(multiband_blend_all(warped_comp, PYRAMID_LEVELS))
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "panorama_multiband.jpg"), blended_mb)

    return naive, blended_lin, blended_mb


def cylindrical_pipeline(images, output_dir):
    """Cylindrical projection + SIFT alignment + multi-band blend."""
    print("  Cylindrical warping...")

    cyl_images = []
    for i, img in enumerate(images):
        cyl = trim_black_borders(cylindrical_warp_vectorized(img))
        cyl_images.append(cyl)
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"cylindrical_{i}.jpg"), cyl)

    kps, descs = extract_sift_features(cyl_images)

    n = len(cyl_images)
    ref = n // 2
    homographies = [None] * n
    homographies[ref] = np.eye(3)

    for i in range(ref - 1, -1, -1):
        good = match_features(descs[i], descs[i + 1], kps[i], kps[i + 1],
                              pair_label=f"cyl_{i}-{i+1}")
        H, _ = compute_homography(kps[i], kps[i + 1], good)
        homographies[i] = (homographies[i + 1] @ H if H is not None
                           else homographies[i + 1].copy())

    for i in range(ref + 1, n):
        good = match_features(descs[i], descs[i - 1], kps[i], kps[i - 1],
                              pair_label=f"cyl_{i}-{i-1}")
        H, _ = compute_homography(kps[i], kps[i - 1], good)
        homographies[i] = (homographies[i - 1] @ H if H is not None
                           else homographies[i - 1].copy())

    canvas_size, offset = compute_canvas_size(cyl_images, homographies)
    T = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]],
                 dtype=np.float64)

    warped = [warp_image(img, T @ H, canvas_size)
              for img, H in zip(cyl_images, homographies)]

    return crop_black(multiband_blend_all(warped))


def _superglue_available():
    try:
        import torch
        sg_dir = os.path.join(os.path.dirname(__file__),
                              "SuperGluePretrainedNetwork")
        return os.path.isdir(sg_dir)
    except ImportError:
        return False


def superglue_pipeline(images, output_dir):
    """SuperPoint + SuperGlue matching (requires PyTorch + pretrained weights)."""
    if not _superglue_available():
        print("[SKIP] SuperGlue not available (needs PyTorch + cloned repo).")
        return None
    print("[SKIP] SuperGlue pipeline not implemented in modular version.")
    return None


def main():
    ap = argparse.ArgumentParser(description="Panorama Stitching Pipeline")
    ap.add_argument("--images_dir", default="images/")
    ap.add_argument("--output_dir", default="output/")
    ap.add_argument("--method", default="all",
                    choices=["all", "standard", "multiband",
                             "cylindrical", "superglue"])
    ap.add_argument("--max_dim", type=int, default=MAX_DIMENSION)
    ap.add_argument("--no_crop", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    images = load_images(args.images_dir, args.max_dim)
    results = {}

    if args.method in ("all", "standard", "multiband"):
        naive, lin, mb = standard_pipeline(images, args.output_dir)
        if naive is not None:
            results["Naive Stitch"] = naive
        if lin is not None:
            results["Linear Feathering"] = lin
        if mb is not None:
            results["Multi-Band Blend"] = mb

    if args.method in ("all", "cylindrical"):
        try:
            cyl = cylindrical_pipeline(images, args.output_dir)
            if cyl is not None:
                cv2.imwrite(os.path.join(args.output_dir,
                            "panorama_cylindrical.jpg"), cyl)
                results["Cylindrical + Multi-Band"] = cyl
        except Exception as e:
            print(f"  [WARN] Cylindrical failed: {e}")

    if args.method in ("all", "superglue"):
        sg = superglue_pipeline(images, args.output_dir)
        if sg is not None:
            results["SuperGlue"] = sg

    if len(results) > 1:
        generate_comparison(results, args.output_dir)

    print(f"\nDone in {time.time() - t0:.1f}s  ->  {os.path.abspath(args.output_dir)}")
    for f in sorted(os.listdir(args.output_dir)):
        sz = os.path.getsize(os.path.join(args.output_dir, f)) / 1024
        print(f"  {f:40s}  {sz:7.1f} KB")


if __name__ == "__main__":
    main()
