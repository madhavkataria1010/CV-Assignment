"""
src.utils — Image I/O, Gain Compensation, Cropping, Reporting.

Utility functions shared across the pipeline.
"""

import os
import sys
import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import MAX_DIMENSION


def load_images(images_dir: str, max_dim: int = MAX_DIMENSION) -> list:
    """
    Load all images from *images_dir*, sorted alphabetically.
    Optionally resize so the longest side ≤ *max_dim*.
    """
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
        paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    paths = sorted(set(paths))

    if len(paths) < 2:
        print(f"[ERROR] Found {len(paths)} image(s) in "
              f"'{images_dir}'. Need at least 2.")
        sys.exit(1)

    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Could not read '{p}', skipping.")
            continue
        if max_dim > 0:
            h, w = img.shape[:2]
            scale = max_dim / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
        images.append(img)
        print(f"  Loaded: {os.path.basename(p)}  →  "
              f"{img.shape[1]}×{img.shape[0]}")

    if len(images) < 2:
        print("[ERROR] Need at least 2 valid images.")
        sys.exit(1)

    print(f"  Total images loaded: {len(images)}\n")
    return images


def gain_compensate(warped_images, masks=None):
    """
    Compute per-image gain factors to minimise intensity differences in
    overlapping regions.

    Uses a ratio-chain approach along consecutive image pairs, then
    normalises so the geometric mean gain is 1.0.
    """
    n = len(warped_images)
    if masks is not None:
        bin_masks = [m > 0 for m in masks]
    else:
        bin_masks = [(img > 0).any(axis=2) for img in warped_images]

    # Compute per-image grayscale once
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
             for img in warped_images]

    # Build gain chain: gain[i] relative to gain[0]=1
    log_gains = np.zeros(n, dtype=np.float64)
    for i in range(n - 1):
        j = i + 1
        overlap = bin_masks[i] & bin_masks[j]
        n_overlap = overlap.sum()
        if n_overlap < 200:
            # No useful overlap — keep same gain as previous
            continue
        # Per-pixel ratio in overlap (robust via median)
        vals_i = grays[i][overlap]
        vals_j = grays[j][overlap]
        # Avoid dark pixels that are noisy
        bright = (vals_i > 15) & (vals_j > 15)
        if bright.sum() < 100:
            continue
        ratio = np.median(vals_j[bright] / vals_i[bright])
        # ratio = median(I_j / I_i) → to match, multiply I_i by ratio
        # Equivalently: gain_j = gain_i / ratio
        log_gains[j] = log_gains[i] - np.log(ratio)

    # Convert to linear gains and normalise so geometric mean = 1
    gains = np.exp(log_gains)
    gains /= np.exp(np.mean(np.log(gains)))  # geomean = 1

    # Soft clip to prevent extreme corrections
    gains = np.clip(gains, 0.6, 1.6)
    # Re-normalise after clipping
    gains /= np.exp(np.mean(np.log(gains)))
    print(f"  Gain factors: {np.round(gains, 3)}")

    return [
        np.clip(img.astype(np.float64) * g, 0, 255).astype(np.uint8)
        for img, g in zip(warped_images, gains)
    ]


def crop_black(img):
    """Crop black borders to produce a clean rectangular panorama."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return img

    all_pts = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_pts)

    margin = 5
    x1 = max(x + margin, 0)
    y1 = max(y + margin, 0)
    x2 = min(x + w - margin, img.shape[1])
    y2 = min(y + h - margin, img.shape[0])
    cropped = img[y1:y2, x1:x2]

    gray_c = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    rows = np.any(gray_c > 2, axis=1)
    cols = np.any(gray_c > 2, axis=0)
    if rows.any() and cols.any():
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        cropped = cropped[ymin:ymax + 1, xmin:xmax + 1]

    return cropped



def generate_comparison(results: dict, output_dir: str):
    """Generate a stacked comparison figure of all stitching methods."""
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(20, 6 * n))
    if n == 1:
        axes = [axes]

    for ax, (label, img) in zip(axes, results.items()):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(label, fontsize=16, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison.jpg")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison saved → {path}\n")
