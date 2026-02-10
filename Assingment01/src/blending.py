"""
src.blending — Image Blending Methods.

Provides three blending strategies (in increasing quality):
  1. Naive overlay        — direct pixel overwrite (creates visible seams).
  2. Linear feathering    — distance-weighted average in overlap regions.
  3. Multi-band blending  — Laplacian-pyramid frequency-domain blend
                            (Burt & Adelson, 1983).
"""

import numpy as np
import cv2

from src.config import PYRAMID_LEVELS


def create_weight_map(shape):
    """
    Distance-based weight map: maximum at the image centre, linearly
    ramping to 0 at every border.
    """
    h, w = shape[:2]
    x = np.linspace(0, 1, w)
    x = np.minimum(x, 1 - x) * 2          # 0 → 1 → 0
    y = np.linspace(0, 1, h)
    y = np.minimum(y, 1 - y) * 2
    return np.outer(y, x).astype(np.float32)


def naive_stitch(warped_images):
    """
    Stack images by simple overwrite: later images paint on top of earlier
    ones.  Produces visible seams wherever intensities differ.
    """
    canvas = np.zeros_like(warped_images[0])
    for warped in warped_images:
        mask = (warped > 0).any(axis=2)
        canvas[mask] = warped[mask]
    return canvas



def linear_blend(warped_images):
    """
    Blend using distance-based weight maps.  In overlap regions the result
    is a weighted average, producing smoother transitions than naive overlay.

        result(x,y) = Σ wᵢ(x,y) · Iᵢ(x,y)  /  Σ wᵢ(x,y)
    """
    canvas     = np.zeros(warped_images[0].shape, dtype=np.float64)
    weight_sum = np.zeros(warped_images[0].shape[:2], dtype=np.float64)

    for warped in warped_images:
        mask  = (warped > 0).any(axis=2).astype(np.float32)
        w_map = create_weight_map(warped.shape) * mask
        w_map = np.maximum(w_map, mask * 1e-6)

        for c in range(3):
            canvas[:, :, c] += warped[:, :, c].astype(np.float64) * w_map
        weight_sum += w_map

    weight_sum = np.maximum(weight_sum, 1e-8)
    for c in range(3):
        canvas[:, :, c] /= weight_sum

    return np.clip(canvas, 0, 255).astype(np.uint8)



def _gaussian_pyramid(img, levels):
    """Build a Gaussian pyramid (list of progressively blurred/downsampled images)."""
    pyramid = [img.astype(np.float64)]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img.astype(np.float64))
    return pyramid


def _laplacian_pyramid(img, levels):
    """
    Build a Laplacian pyramid.

    Each level  L_l = G_l − upsample(G_{l+1})  captures band-pass detail
    at that resolution.  The coarsest level is the low-pass residual.
    """
    gp = _gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels - 1):
        h, w = gp[i].shape[:2]
        upsampled = cv2.pyrUp(gp[i + 1], dstsize=(w, h))
        lp.append(gp[i] - upsampled)
    lp.append(gp[-1])
    return lp


def _collapse_pyramid(lp):
    """Reconstruct an image from its Laplacian pyramid."""
    img = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        h, w = lp[i].shape[:2]
        img = cv2.pyrUp(img, dstsize=(w, h)) + lp[i]
    return img


def multiband_blend_pair(img_a, img_b, mask_a, mask_b,
                         levels=PYRAMID_LEVELS):
    """Blend two overlapping images using Laplacian pyramid multi-band blending."""
    img_a = img_a.astype(np.float64)
    img_b = img_b.astype(np.float64)

    # 3-channel masks
    if mask_a.ndim == 2:
        mask_a_3 = np.stack([mask_a] * 3, axis=-1).astype(np.float64)
    else:
        mask_a_3 = mask_a.astype(np.float64)
    if mask_b.ndim == 2:
        mask_b_3 = np.stack([mask_b] * 3, axis=-1).astype(np.float64)
    else:
        mask_b_3 = mask_b.astype(np.float64)

    # Build blend mask: 1 where only A, 0 where only B, smooth in overlap
    overlap  = (mask_a_3 > 0) & (mask_b_3 > 0)
    only_a   = (mask_a_3 > 0) & (mask_b_3 == 0)
    only_b   = (mask_a_3 == 0) & (mask_b_3 > 0)

    blend_mask = np.zeros_like(mask_a_3)
    blend_mask[only_a] = 1.0
    blend_mask[only_b] = 0.0

    # In overlap: weight by distance from centre
    w_a   = create_weight_map(img_a.shape)
    w_b   = create_weight_map(img_b.shape)
    w_a_3 = np.stack([w_a] * 3, axis=-1) * mask_a_3
    w_b_3 = np.stack([w_b] * 3, axis=-1) * mask_b_3
    total  = np.maximum(w_a_3 + w_b_3, 1e-8)

    for c in range(3):
        ov = overlap[:, :, c]
        blend_mask[:, :, c][ov] = w_a_3[:, :, c][ov] / total[:, :, c][ov]

    # Pyramids
    la = _laplacian_pyramid(img_a, levels)
    lb = _laplacian_pyramid(img_b, levels)
    gm = _gaussian_pyramid(blend_mask, levels)

    blended = []
    for la_l, lb_l, gm_l in zip(la, lb, gm):
        min_h = min(la_l.shape[0], lb_l.shape[0], gm_l.shape[0])
        min_w = min(la_l.shape[1], lb_l.shape[1], gm_l.shape[1])
        la_l = la_l[:min_h, :min_w]
        lb_l = lb_l[:min_h, :min_w]
        gm_l = gm_l[:min_h, :min_w]
        blended.append(gm_l * la_l + (1.0 - gm_l) * lb_l)

    result = _collapse_pyramid(blended)
    return np.clip(result, 0, 255).astype(np.uint8)


def multiband_blend_all(warped_images, levels=PYRAMID_LEVELS):
    """Iteratively blend a list of warped images via multi-band blending."""
    result = warped_images[0]
    for i in range(1, len(warped_images)):
        mask_r = (result > 0).any(axis=2).astype(np.float32)
        mask_i = (warped_images[i] > 0).any(axis=2).astype(np.float32)
        result = multiband_blend_pair(result, warped_images[i],
                                      mask_r, mask_i, levels)
    return result
