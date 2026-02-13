"""
src.evaluation — Quantitative Evaluation Metrics for Panorama Stitching.

Provides metrics to compare blending methods:
  1. PSNR  — Peak Signal-to-Noise Ratio in overlap regions
  2. SSIM  — Structural Similarity Index in overlap regions
  3. Seam  — Mean gradient magnitude along seam (lower = smoother blend)
  4. Edge  — Laplacian variance (sharpness / detail preservation)
  5. Colour Consistency — std-dev of mean intensity across vertical strips
  6. Processing Time — wall-clock time per method
"""

import time
import numpy as np
import cv2


# ── helpers ──────────────────────────────────────────────────────────────

def _overlap_mask(warped_images):
    """Return a binary mask of pixels covered by ≥ 2 warped images."""
    count = np.zeros(warped_images[0].shape[:2], dtype=np.int32)
    for w in warped_images:
        count += (w > 0).any(axis=2).astype(np.int32)
    return count >= 2


def _seam_mask(warped_images, thickness=3):
    """
    Approximate the seam zone: the boundary between the coverage masks of
    consecutive image pairs.  Returns a binary mask dilated to *thickness*.
    """
    seam = np.zeros(warped_images[0].shape[:2], dtype=np.uint8)
    for i in range(len(warped_images) - 1):
        m1 = (warped_images[i] > 0).any(axis=2).astype(np.uint8)
        m2 = (warped_images[i + 1] > 0).any(axis=2).astype(np.uint8)
        overlap = m1 & m2
        if overlap.sum() == 0:
            continue
        # Boundary of overlap region ≈ seam
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge = cv2.morphologyEx(overlap, cv2.MORPH_GRADIENT, kernel)
        seam = cv2.bitwise_or(seam, edge)
    # Dilate to get a strip around the seam
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (thickness * 2 + 1, thickness * 2 + 1))
    seam = cv2.dilate(seam, kernel)
    return seam.astype(bool)


def _weighted_reference(warped_images):
    """
    Build a pixel-wise weighted-average "ground-truth" reference from the
    warped source images (using distance-transform weights).
    Used to compute PSNR / SSIM of each blending result in the overlap.
    """
    from src.blending import _distance_weight

    canvas = np.zeros(warped_images[0].shape, dtype=np.float64)
    wsum = np.zeros(warped_images[0].shape[:2], dtype=np.float64)
    for w in warped_images:
        mask = (w > 0).any(axis=2).astype(np.uint8) * 255
        wm = _distance_weight(mask)
        wm = np.maximum(wm, (mask > 0).astype(np.float32) * 1e-6)
        for c in range(3):
            canvas[:, :, c] += w[:, :, c].astype(np.float64) * wm
        wsum += wm
    wsum = np.maximum(wsum, 1e-8)
    for c in range(3):
        canvas[:, :, c] /= wsum
    return np.clip(canvas, 0, 255).astype(np.uint8)


# ── individual metrics ───────────────────────────────────────────────────

def psnr_overlap(result, warped_images):
    """
    PSNR between the blended result and the weighted-average reference
    image, computed only in the overlap region.  Higher is better.
    """
    ref = _weighted_reference(warped_images)
    mask = _overlap_mask(warped_images)
    if mask.sum() == 0:
        return float("nan")

    r_pixels = result[mask].astype(np.float64)
    f_pixels = ref[mask].astype(np.float64)
    mse = np.mean((r_pixels - f_pixels) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_overlap(result, warped_images):
    """
    Mean SSIM between the blended result and the weighted-average
    reference, computed only inside the overlap region.  Higher is better.
    """
    ref = _weighted_reference(warped_images)
    mask = _overlap_mask(warped_images)
    if mask.sum() == 0:
        return float("nan")

    # Compute full SSIM map, then average inside the overlap
    gray_r = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_f = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    k = 11  # window size
    window = cv2.getGaussianKernel(k, 1.5)
    window = window @ window.T

    mu_r = cv2.filter2D(gray_r, -1, window)
    mu_f = cv2.filter2D(gray_f, -1, window)

    mu_r_sq = mu_r ** 2
    mu_f_sq = mu_f ** 2
    mu_rf = mu_r * mu_f

    sigma_r_sq = cv2.filter2D(gray_r ** 2, -1, window) - mu_r_sq
    sigma_f_sq = cv2.filter2D(gray_f ** 2, -1, window) - mu_f_sq
    sigma_rf = cv2.filter2D(gray_r * gray_f, -1, window) - mu_rf

    ssim_map = ((2 * mu_rf + C1) * (2 * sigma_rf + C2)) / \
               ((mu_r_sq + mu_f_sq + C1) * (sigma_r_sq + sigma_f_sq + C2))

    # Erode mask to avoid border artefacts from filtering
    mask_eroded = cv2.erode(mask.astype(np.uint8),
                            np.ones((k, k), np.uint8)).astype(bool)
    if mask_eroded.sum() == 0:
        mask_eroded = mask
    return float(np.mean(ssim_map[mask_eroded]))


def seam_visibility(result, warped_images, thickness=3):
    """
    Mean gradient magnitude of the blended result along the seam zone.
    Lower value  ⇒  smoother, less visible seam.
    """
    smask = _seam_mask(warped_images, thickness)
    if smask.sum() == 0:
        return float("nan")

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(mag[smask]))


def edge_preservation(result, warped_images):
    """
    Laplacian variance of the result in the content region (non-black).
    Higher value  ⇒  sharper / more detail preserved.
    """
    content = (result > 0).any(axis=2)
    if content.sum() == 0:
        return float("nan")
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap[content]))


def colour_consistency(result):
    """
    Divide the panorama into vertical strips and measure the standard
    deviation of per-strip mean intensity.  Lower  ⇒  more consistent
    colour across the panorama.
    """
    content = (result > 0).any(axis=2)
    if content.sum() == 0:
        return float("nan")

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float64)
    n_strips = 10
    h, w = gray.shape
    strip_w = w // n_strips
    if strip_w == 0:
        return float("nan")

    means = []
    for s in range(n_strips):
        x0 = s * strip_w
        x1 = x0 + strip_w
        region = gray[:, x0:x1]
        mask_region = content[:, x0:x1]
        if mask_region.sum() > 0:
            means.append(np.mean(region[mask_region]))

    if len(means) < 2:
        return float("nan")
    return float(np.std(means))


# ── aggregate evaluation ─────────────────────────────────────────────────

def evaluate_all(results: dict, warped_images: list,
                 timings: dict | None = None):
    """
    Run every metric on each blending result.

    Parameters
    ----------
    results : dict[str, np.ndarray]
        Mapping  method_name → blended BGR panorama.
    warped_images : list[np.ndarray]
        The warped source images (before blending).
    timings : dict[str, float] | None
        Optional mapping  method_name → seconds.

    Returns
    -------
    metrics : dict[str, dict[str, float]]
        Nested dict   method → metric → value.
    """
    metrics = {}
    for name, img in results.items():
        print(f"  Evaluating '{name}' …")
        # Resize warped images to match result if crop_black changed size
        # We need unscropped warped images — use them at canvas resolution
        m = {
            "PSNR (dB) ↑":            psnr_overlap(img, warped_images),
            "SSIM ↑":                 ssim_overlap(img, warped_images),
            "Seam Gradient ↓":        seam_visibility(img, warped_images),
            "Edge Variance ↑":        edge_preservation(img, warped_images),
            "Colour Consistency ↓":   colour_consistency(img),
        }
        if timings and name in timings:
            m["Time (s)"] = timings[name]
        metrics[name] = m
    return metrics


def print_metrics_table(metrics: dict):
    """Pretty-print the metrics as a table to stdout."""
    methods = list(metrics.keys())
    if not methods:
        return
    metric_names = list(metrics[methods[0]].keys())

    col_w = max(len(m) for m in methods) + 2
    hdr = f"{'Metric':<28s}" + "".join(f"{m:>{col_w}s}" for m in methods)
    print("\n" + "=" * len(hdr))
    print("  EVALUATION METRICS")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for mn in metric_names:
        row = f"{mn:<28s}"
        for m in methods:
            v = metrics[m].get(mn, float("nan"))
            if mn == "Time (s)":
                row += f"{v:>{col_w}.3f}"
            elif mn.startswith("SSIM"):
                row += f"{v:>{col_w}.4f}"
            else:
                row += f"{v:>{col_w}.2f}"
        print(row)
    print("=" * len(hdr) + "\n")


def save_metrics_csv(metrics: dict, path: str):
    """Write metrics to a CSV file for easy inclusion in reports."""
    methods = list(metrics.keys())
    if not methods:
        return
    metric_names = list(metrics[methods[0]].keys())

    with open(path, "w") as f:
        f.write("Metric," + ",".join(methods) + "\n")
        for mn in metric_names:
            vals = ",".join(
                f"{metrics[m].get(mn, float('nan')):.6f}" for m in methods
            )
            f.write(f"{mn},{vals}\n")
    print(f"  Metrics saved → {path}")
