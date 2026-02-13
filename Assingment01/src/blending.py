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


def _distance_weight(mask):
    """
    Compute a distance-transform weight map from a binary mask.

    Uses raw (unnormalised) distance values so that the ratio between
    two images in an overlap region naturally reflects proximity to
    each image's boundary.  Squaring the distances sharpens the
    preference for pixels deep inside the content region, which
    reduces ghosting in the overlap zone.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Square the distance to give stronger preference to center pixels
    dist = dist ** 2
    return dist.astype(np.float32)


def create_weight_map(shape):
    """
    Legacy distance-based weight map (kept for API compatibility).
    Prefer _distance_weight(mask) for warped images.
    """
    h, w = shape[:2]
    x = np.linspace(0, 1, w)
    x = np.minimum(x, 1 - x) * 2
    y = np.linspace(0, 1, h)
    y = np.minimum(y, 1 - y) * 2
    return np.outer(y, x).astype(np.float32)


def naive_stitch(warped_images, masks=None):
    """
    Stack images by simple overwrite: later images paint on top of earlier
    ones.  Produces visible seams wherever intensities differ.
    """
    canvas = np.zeros_like(warped_images[0])
    for i, warped in enumerate(warped_images):
        if masks is not None:
            mask = masks[i] > 0
        else:
            mask = (warped > 0).any(axis=2)
        canvas[mask] = warped[mask]
    return canvas



def _linear_weighted_average(warped_images, masks=None):
    """
    Blend using distance-transform weight maps.  In overlap regions the
    result is a weighted average, producing smoother transitions.

        result(x,y) = Σ wᵢ(x,y) · Iᵢ(x,y)  /  Σ wᵢ(x,y)
    """
    canvas     = np.zeros(warped_images[0].shape, dtype=np.float64)
    weight_sum = np.zeros(warped_images[0].shape[:2], dtype=np.float64)

    for i, warped in enumerate(warped_images):
        if masks is not None:
            m = masks[i]
        else:
            m = ((warped > 0).any(axis=2).astype(np.uint8) * 255)
        w_map = _distance_weight(m)
        w_map = np.maximum(w_map, (m > 0).astype(np.float32) * 1e-6)

        for c in range(3):
            canvas[:, :, c] += warped[:, :, c].astype(np.float64) * w_map
        weight_sum += w_map

    weight_sum = np.maximum(weight_sum, 1e-8)
    for c in range(3):
        canvas[:, :, c] /= weight_sum

    return np.clip(canvas, 0, 255).astype(np.uint8)


def _label_seam_blend(warped_images, masks, feather_sigma=6.0, center_boost=1.12):
    """
    Deghost blend via label ownership + narrow feather.

    Each output pixel is assigned to the source image with maximum
    distance-to-boundary weight, then softly feathered near seam lines.
    """
    n = len(warped_images)
    if n == 1:
        return warped_images[0].copy()

    masks_u8 = [_mask_uint8(m) for m in masks]
    weights = []
    for m in masks_u8:
        w = _distance_weight(m)
        w = np.maximum(w, (m > 0).astype(np.float32) * 1e-6)
        weights.append(w)

    w_stack = np.stack(weights, axis=0)
    ref = n // 2
    w_stack[ref] *= center_boost

    labels = np.argmax(w_stack, axis=0)
    valid = w_stack.max(axis=0) > 0

    soft = []
    for i in range(n):
        m = ((labels == i) & valid).astype(np.float32)
        if feather_sigma > 0:
            m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
        m *= (masks_u8[i] > 0).astype(np.float32)
        soft.append(m)

    s = np.stack(soft, axis=0)
    s_sum = np.maximum(s.sum(axis=0), 1e-6)
    s /= s_sum

    out = np.zeros_like(warped_images[0], dtype=np.float32)
    for i, img in enumerate(warped_images):
        out += img.astype(np.float32) * s[i][:, :, None]

    return np.clip(out, 0, 255).astype(np.uint8)


def linear_blend(warped_images, masks=None):
    """
    Blend panoramas with adaptive deghosting.

    Uses classic weighted averaging for easy scenes and switches to
    label-seam blending when overlap disagreement is high.
    """
    if masks is not None:
        try:
            if _overlap_disagreement(warped_images, masks) >= 18.0:
                return _label_seam_blend(warped_images, masks)
        except Exception as exc:
            print(f"  [WARN] Deghost linear fallback: {exc}")
    return _linear_weighted_average(warped_images, masks)



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

    # Ensure masks are 2-D uint8
    if mask_a.ndim == 3:
        mask_a = mask_a[:, :, 0]
    if mask_b.ndim == 3:
        mask_b = mask_b[:, :, 0]
    mask_a = (mask_a > 0).astype(np.uint8) * 255
    mask_b = (mask_b > 0).astype(np.uint8) * 255

    # Distance-transform weights (unnormalised, squared for sharper falloff)
    w_a = _distance_weight(mask_a)
    w_b = _distance_weight(mask_b)

    # Build smooth blend mask: ratio of distance weights
    total = np.maximum(w_a + w_b, 1e-8)
    blend_2d = w_a / total

    # Where only one image exists, force 0 or 1
    only_a = (mask_a > 0) & (mask_b == 0)
    only_b = (mask_a == 0) & (mask_b > 0)
    blend_2d[only_a] = 1.0
    blend_2d[only_b] = 0.0

    # Smooth the blend mask to prevent hard transitions
    ksize = max(3, min(blend_2d.shape) // 8)
    if ksize % 2 == 0:
        ksize += 1
    blend_2d = cv2.GaussianBlur(blend_2d, (ksize, ksize), 0)

    # Expand to 3 channels
    blend_mask = np.stack([blend_2d] * 3, axis=-1).astype(np.float64)

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


def _mask_uint8(mask):
    """Convert any mask format to single-channel uint8 {0,255}."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8) * 255


def _extract_roi_payloads(warped_images, masks):
    """Extract tight ROIs for each warped image and its mask."""
    payloads = []
    for img, mask in zip(warped_images, masks):
        m = _mask_uint8(mask)
        ys, xs = np.where(m > 0)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        payloads.append({
            "img": img[y0:y1, x0:x1],
            "mask": m[y0:y1, x0:x1],
            "corner": (x0, y0),
        })
    return payloads


def _find_seam_masks(payloads):
    """Find seam masks using GraphCut, with DP fallback."""
    if len(payloads) <= 1:
        return [p["mask"].copy() for p in payloads]

    imgs32 = [p["img"].astype(np.float32) for p in payloads]
    corners = [p["corner"] for p in payloads]
    seam_masks = [p["mask"].copy() for p in payloads]

    try:
        seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        seam_finder.find(imgs32, corners, seam_masks)
    except Exception:
        seam_finder = cv2.detail_DpSeamFinder("COLOR_GRAD")
        seam_finder.find(imgs32, corners, seam_masks)

    # Guard against pathological empty seams
    for i, p in enumerate(payloads):
        if np.count_nonzero(seam_masks[i]) == 0:
            seam_masks[i] = p["mask"].copy()
    return seam_masks


def _multiband_blend_detail(warped_images, masks, levels):
    """
    OpenCV detail-module multi-band blend with seam masks.

    This reduces ghosting in overlaps by cutting seams through
    low-cost regions before pyramid blending.
    """
    payloads = _extract_roi_payloads(warped_images, masks)
    if not payloads:
        return np.zeros_like(warped_images[0])
    if len(payloads) == 1:
        out = np.zeros_like(warped_images[0])
        p = payloads[0]
        x0, y0 = p["corner"]
        h, w = p["img"].shape[:2]
        out[y0:y0 + h, x0:x0 + w][p["mask"] > 0] = p["img"][p["mask"] > 0]
        return out

    seam_masks = _find_seam_masks(payloads)

    canvas_h, canvas_w = warped_images[0].shape[:2]
    blender = cv2.detail_MultiBandBlender(0, int(max(1, levels)))
    blender.prepare((0, 0, canvas_w, canvas_h))

    for p, seam_mask in zip(payloads, seam_masks):
        blend_mask = cv2.bitwise_and(p["mask"], seam_mask)
        if np.count_nonzero(blend_mask) == 0:
            blend_mask = p["mask"]
        blender.feed(p["img"].astype(np.int16), blend_mask, p["corner"])

    result, _ = blender.blend(None, None)
    return np.clip(result, 0, 255).astype(np.uint8)


def _overlap_disagreement(warped_images, masks):
    """
    Estimate photometric disagreement in overlap zones.

    Higher values usually indicate parallax/misalignment where seam cuts
    are preferable to wide smooth blending.
    """
    diffs = []
    n = len(warped_images)
    masks_u8 = [_mask_uint8(m) for m in masks]
    for i in range(n):
        for j in range(i + 1, n):
            overlap = (masks_u8[i] > 0) & (masks_u8[j] > 0)
            if overlap.sum() < 2000:
                continue
            ai = warped_images[i][overlap].astype(np.float32)
            bj = warped_images[j][overlap].astype(np.float32)
            diffs.append(float(np.mean(np.abs(ai - bj))))
    if not diffs:
        return 0.0
    return float(np.median(diffs))


def multiband_blend_all(warped_images, levels=PYRAMID_LEVELS, masks=None):
    """Iteratively blend a list of warped images via multi-band blending.

    Blends from edges inward (outermost images first, then toward center)
    to minimize error accumulation.
    """
    n = len(warped_images)
    if n == 1:
        return warped_images[0].copy()

    # Prefer seam-aware global blending when masks are available.
    # Fallback to the legacy pairwise routine if the detail module path fails.
    if masks is not None and hasattr(cv2, "detail_MultiBandBlender"):
        try:
            disagreement = _overlap_disagreement(warped_images, masks)
            # Very hard parallax: use stronger deghost ownership blend.
            if disagreement >= 26.0:
                return _label_seam_blend(warped_images, masks)
            # Use seam cuts only when overlap disagreement is high
            # (strong parallax / ghosting risk).
            if disagreement >= 22.0:
                return _multiband_blend_detail(warped_images, masks, levels)
        except Exception as exc:
            print(f"  [WARN] Seam-aware blending fallback: {exc}")

    ref = n // 2

    # Start with center image
    result = warped_images[ref].copy()
    if masks is not None:
        result_mask = masks[ref].copy()
    else:
        result_mask = ((result > 0).any(axis=2).astype(np.uint8) * 255)

    # Interleave left and right from center outward
    order = []
    for d in range(1, n):
        if ref - d >= 0:
            order.append(ref - d)
        if ref + d < n:
            order.append(ref + d)

    for i in order:
        if masks is not None:
            mask_i = masks[i]
        else:
            mask_i = ((warped_images[i] > 0).any(axis=2).astype(np.uint8) * 255)

        result = multiband_blend_pair(result, warped_images[i],
                                      result_mask, mask_i, levels)
        # Update the combined mask
        result_mask = np.maximum(result_mask, mask_i)
    return result
