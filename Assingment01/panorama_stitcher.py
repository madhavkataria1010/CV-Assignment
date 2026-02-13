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
    cylindrical_warp_vectorized, trim_black_borders, refine_alignment_ecc,
)
from src.blending import naive_stitch, linear_blend, multiband_blend_all
from src.utils import load_images, gain_compensate, crop_black, generate_comparison
from src.evaluation import evaluate_all, print_metrics_table, save_metrics_csv


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
            return None, None, None, None, None, None
        homographies[i] = homographies[i + 1] @ H

    for i in range(ref + 1, n):
        good = match_features(descs[i], descs[i - 1], kps[i], kps[i - 1],
                              images[i], images[i - 1], output_dir,
                              pair_label=f"{i}-{i-1}")
        H, _ = compute_homography(kps[i], kps[i - 1], good)
        if H is None:
            print(f"  [ERROR] Cannot stitch image {i}.")
            return None, None, None, None, None, None
        homographies[i] = homographies[i - 1] @ H

    canvas_size, offset = compute_canvas_size(images, homographies)
    T = np.array([[1, 0, offset[0]],
                  [0, 1, offset[1]],
                  [0, 0, 1]], dtype=np.float64)

    warped_images = []
    warp_masks = []
    for i, (img, H) in enumerate(zip(images, homographies)):
        w, m = warp_image(img, T @ H, canvas_size)
        warped_images.append(w)
        warp_masks.append(m)
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"warped_{i}.jpg"), w)

    # Post-warp ECC refinement against the reference frame reduces
    # residual misalignment that causes visible ghosting.
    warped_images, warp_masks = refine_alignment_ecc(warped_images, warp_masks, ref)

    if output_dir:
        for i, w in enumerate(warped_images):
            cv2.imwrite(os.path.join(output_dir, f"warped_{i}.jpg"), w)

    t1 = time.time()
    naive = naive_stitch(warped_images, warp_masks)
    t_naive = time.time() - t1
    naive_cropped = crop_black(naive)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "naive_stitch.jpg"), naive_cropped)

    warped_comp = gain_compensate(warped_images, warp_masks)

    t1 = time.time()
    blended_lin = linear_blend(warped_comp, warp_masks)
    t_lin = time.time() - t1
    blended_lin_cropped = crop_black(blended_lin)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "panorama_linear.jpg"), blended_lin_cropped)

    t1 = time.time()
    blended_mb = multiband_blend_all(warped_comp, PYRAMID_LEVELS, warp_masks)
    t_mb = time.time() - t1
    blended_mb_cropped = crop_black(blended_mb)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "panorama_multiband.jpg"), blended_mb_cropped)

    timings = {"Naive Stitch": t_naive, "Linear Feathering": t_lin,
               "Multi-Band Blend": t_mb}

    # Return cropped versions for display AND uncropped for evaluation
    uncropped = {"Naive Stitch": naive, "Linear Feathering": blended_lin,
                 "Multi-Band Blend": blended_mb}

    return (naive_cropped, blended_lin_cropped, blended_mb_cropped,
            warped_comp, timings, uncropped)


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

    warped_pairs = [warp_image(img, T @ H, canvas_size)
                    for img, H in zip(cyl_images, homographies)]
    warped = [p[0] for p in warped_pairs]
    cyl_masks = [p[1] for p in warped_pairs]

    return crop_black(multiband_blend_all(warped, masks=cyl_masks))


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

    timings = {}
    warped_for_eval = None
    uncropped = {}

    if args.method in ("all", "standard", "multiband"):
        pipeline_out = standard_pipeline(images, args.output_dir)
        if pipeline_out[0] is not None:
            naive, lin, mb, warped_for_eval, blend_timings, uncropped = pipeline_out
            results["Naive Stitch"] = naive
            results["Linear Feathering"] = lin
            results["Multi-Band Blend"] = mb
            timings.update(blend_timings)

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

    # ── Evaluation ────────────────────────────────────────────────────
    if warped_for_eval is not None and len(results) >= 1:
        eval_results = {k: v for k, v in uncropped.items()
                        if k in results}
        metrics = evaluate_all(eval_results, warped_for_eval, timings)
        print_metrics_table(metrics)
        csv_path = os.path.join(args.output_dir, "metrics.csv")
        save_metrics_csv(metrics, csv_path)

    print(f"\nDone in {time.time() - t0:.1f}s  ->  {os.path.abspath(args.output_dir)}")
    for f in sorted(os.listdir(args.output_dir)):
        sz = os.path.getsize(os.path.join(args.output_dir, f)) / 1024
        print(f"  {f:40s}  {sz:7.1f} KB")


if __name__ == "__main__":
    main()
