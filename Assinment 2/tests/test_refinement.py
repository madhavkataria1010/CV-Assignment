from __future__ import annotations

import numpy as np

from src.config import RefinementConfig
from src.refinement import refine_mask


def test_refinement_removes_noise_and_preserves_seeds() -> None:
    raw_mask = np.zeros((20, 20), dtype=bool)
    raw_mask[5:15, 5:15] = True
    raw_mask[1, 1] = True
    fg_seed = np.zeros_like(raw_mask)
    bg_seed = np.zeros_like(raw_mask)
    fg_seed[10, 10] = True
    bg_seed[2, 18] = True

    refined = refine_mask(
        raw_mask,
        fg_seed,
        bg_seed,
        RefinementConfig(
            min_object_size=20,
            min_hole_size=20,
            opening_radius=1,
            closing_radius=1,
            smoothing_sigma=0.6,
        ),
    )

    assert refined[10, 10]
    assert not refined[2, 18]
    assert not refined[1, 1]
