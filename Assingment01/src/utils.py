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


def gain_compensate(warped_images):
    """
    Compute per-image gain factors to minimise intensity differences in
    overlapping regions.

    Solves:  argmin_{g_i}  Σ_{i<j}  Σ_{overlap}  (g_i · I_i − g_j · I_j)²

    with constraint  Σ g_i = N  (average gain = 1).
    """
    n = len(warped_images)
    masks = [(img > 0).any(axis=2) for img in warped_images]

    A_rows, b_rows = [], []
    for i in range(n):
        for j in range(i + 1, n):
            overlap = masks[i] & masks[j]
            if overlap.sum() < 100:
                continue
            mean_i = warped_images[i][overlap].mean()
            mean_j = warped_images[j][overlap].mean()
            if mean_i < 1 or mean_j < 1:
                continue
            row = np.zeros(n)
            row[i]  =  mean_i
            row[j]  = -mean_j
            A_rows.append(row)
            b_rows.append(0.0)

    if len(A_rows) < 1:
        return warped_images

    A_rows.append(np.ones(n))
    b_rows.append(float(n))

    A = np.array(A_rows)
    b = np.array(b_rows)
    gains, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    gains = np.clip(gains, 0.5, 2.0)
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
