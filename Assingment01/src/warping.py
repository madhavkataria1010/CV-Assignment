"""
src.warping — Image Warping & Cylindrical Projection.

Handles:
  • Computing the output canvas bounding box from warped corners.
  • Perspective warping via cv2.warpPerspective.
  • Cylindrical projection (novel method) for reduced edge distortion.
"""

import numpy as np
import cv2


def compute_canvas_size(images, homographies):
    """
    Determine the output canvas by warping each image's four corners through
    its homography and computing the axis-aligned bounding box.

    Returns
    -------
    canvas_size : (width, height)
    offset      : (x_offset, y_offset)  — translation for positive coords
    """
    all_corners = []
    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        corners = np.float32(
            [[0, 0], [w, 0], [w, h], [0, h]]
        ).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped_corners)

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    offset      = (-x_min, -y_min)
    canvas_size = (x_max - x_min, y_max - y_min)
    return canvas_size, offset


def warp_image(img, H, canvas_size):
    """Warp a single image onto the output canvas using homography *H*.

    Returns
    -------
    warped : np.ndarray — the warped BGR image
    mask   : np.ndarray — uint8 binary mask (255 where content exists)
    """
    warped = cv2.warpPerspective(
        img, H, canvas_size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Warp a white image to get a precise content mask
    white = np.ones(img.shape[:2], dtype=np.uint8) * 255
    mask = cv2.warpPerspective(
        white, H, canvas_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Erode mask slightly to remove interpolation artifacts at borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=1)
    # Zero out warped pixels outside the eroded mask
    warped[mask == 0] = 0
    return warped, mask


def refine_alignment_ecc(warped_images, masks, ref_index,
                         motion=cv2.MOTION_AFFINE,
                         min_overlap=5000,
                         max_shift=40.0):
    """
    Refine warped image alignment to the reference image using ECC.

    This is a local post-warp correction step that helps reduce
    ghosting when global homographies are imperfect.
    """
    n = len(warped_images)
    if n <= 1:
        return warped_images, masks

    out_images = [img.copy() for img in warped_images]
    out_masks = [m.copy() for m in masks]

    ref_gray = cv2.cvtColor(out_images[ref_index], cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        120,
        1e-6,
    )

    for i in range(n):
        if i == ref_index:
            continue

        overlap = ((out_masks[i] > 0) & (out_masks[ref_index] > 0)).astype(np.uint8)
        if int(overlap.sum()) < min_overlap:
            continue

        img_gray = cv2.cvtColor(out_images[i], cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                ref_gray,
                img_gray,
                warp,
                motion,
                criteria,
                inputMask=overlap,
                gaussFiltSize=5,
            )
        except cv2.error:
            continue

        # Reject unstable solutions.
        tx, ty = float(warp[0, 2]), float(warp[1, 2])
        if abs(tx) > max_shift or abs(ty) > max_shift:
            continue

        A = warp[:, :2]
        det = float(np.linalg.det(A))
        if det < 0.85 or det > 1.15:
            continue

        h, w = out_images[i].shape[:2]
        out_images[i] = cv2.warpAffine(
            out_images[i], warp, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out_masks[i] = cv2.warpAffine(
            out_masks[i], warp, (w, h),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return out_images, out_masks


def cylindrical_warp_vectorized(img, focal_length=None):
    """Project an image onto a cylindrical surface (vectorised).

    Inverse map:  theta = (xc - cx) / f
                  xs = f * tan(theta) + cx
                  ys = (yc - cy) / cos(theta) + cy

    focal_length defaults to image width (~53 deg FOV).
    """
    h, w = img.shape[:2]
    if focal_length is None:
        focal_length = float(w)

    cx, cy = w / 2.0, h / 2.0
    f = focal_length

    xc = np.arange(w, dtype=np.float32)
    yc = np.arange(h, dtype=np.float32)
    xc, yc = np.meshgrid(xc, yc)

    theta = (xc - cx) / f
    xs = f * np.tan(theta) + cx
    ys = (yc - cy) / np.cos(theta) + cy

    warped = cv2.remap(
        img,
        xs.astype(np.float32),
        ys.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped


def trim_black_borders(img):
    """Trim rows/columns that are predominantly black after warping."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    row_sum = thresh.sum(axis=1)
    col_sum = thresh.sum(axis=0)

    row_mask = row_sum > thresh.shape[1] * 0.1 * 255
    col_mask = col_sum > thresh.shape[0] * 0.1 * 255

    rows = np.where(row_mask)[0]
    cols = np.where(col_mask)[0]

    if len(rows) == 0 or len(cols) == 0:
        return img

    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
