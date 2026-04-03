from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

from src.config import RefinementConfig


def _keep_seed_connected_components(mask: np.ndarray, fg_seed: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask
    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(labels[fg_seed])] = True
    keep[0] = False
    return keep[labels]


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max(1, min_size)
    keep[0] = False
    return keep[labels]


def _fill_small_holes(mask: np.ndarray, max_hole_size: int) -> np.ndarray:
    holes = ~mask
    labels, count = ndi.label(holes)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    fill = sizes <= max(1, max_hole_size)
    fill[0] = False
    result = mask.copy()
    result[fill[labels]] = True
    return result


def refine_mask(
    raw_mask: np.ndarray,
    fg_seed: np.ndarray,
    bg_seed: np.ndarray,
    config: RefinementConfig,
) -> np.ndarray:
    refined = raw_mask.astype(bool)
    refined = _remove_small_components(refined, config.min_object_size)
    refined = ndi.binary_opening(refined, structure=disk(config.opening_radius))
    refined = ndi.binary_closing(refined, structure=disk(config.closing_radius))
    refined = _fill_small_holes(refined, config.min_hole_size)
    refined = ndi.gaussian_filter(refined.astype(np.float32), sigma=config.smoothing_sigma) >= 0.5
    refined = _keep_seed_connected_components(refined, fg_seed)
    refined[fg_seed] = True
    refined[bg_seed] = False
    return refined.astype(bool)
