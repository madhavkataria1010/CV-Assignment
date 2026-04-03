from __future__ import annotations

import numpy as np


def run_naive_segmentation(
    fg_cost: np.ndarray,
    bg_cost: np.ndarray,
    fg_seed: np.ndarray,
    bg_seed: np.ndarray,
) -> np.ndarray:
    mask = fg_cost <= bg_cost
    mask[fg_seed] = True
    mask[bg_seed] = False
    return mask.astype(bool)

