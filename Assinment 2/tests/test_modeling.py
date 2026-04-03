from __future__ import annotations

import numpy as np

from src.modeling import compute_unary_costs, initialize_models, rgb_to_lab


def test_histogram_model_prefers_seeded_class_colors() -> None:
    image = np.array(
        [
            [[255, 0, 0], [255, 0, 0]],
            [[0, 0, 255], [0, 0, 255]],
        ],
        dtype=np.uint8,
    )
    fg_seed = np.array([[True, False], [False, False]])
    bg_seed = np.array([[False, False], [True, False]])

    fg_model, bg_model = initialize_models(rgb_to_lab(image), fg_seed, bg_seed, (8, 8, 8), 1.0)
    fg_cost, bg_cost = compute_unary_costs(
        rgb_to_lab(image), fg_model, bg_model, fg_seed, bg_seed, hard_seed_cost=1e6
    )

    assert fg_cost[0, 1] < bg_cost[0, 1]
    assert bg_cost[1, 1] < fg_cost[1, 1]
    assert fg_cost[0, 0] == 0.0
    assert bg_cost[1, 0] == 0.0
