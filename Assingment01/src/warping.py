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
    """Warp a single image onto the output canvas using homography *H*."""
    return cv2.warpPerspective(
        img, H, canvas_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


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
