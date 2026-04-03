from __future__ import annotations

import numpy as np

from src.evaluation import bbox_fill_ratio, bbox_leakage_ratio, compactness, edge_alignment_score


def test_bbox_metrics_and_compactness_behave_reasonably() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 3:9] = True
    bbox = (2, 2, 10, 10)

    assert bbox_leakage_ratio(mask, bbox) == 0.0
    assert 0.0 < bbox_fill_ratio(mask, bbox) < 1.0
    assert compactness(mask) > 0.0


def test_edge_alignment_score_prefers_boundary_on_strong_edges() -> None:
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[:, 6:] = 255

    aligned = np.zeros((12, 12), dtype=bool)
    aligned[:, :6] = True

    misaligned = np.zeros((12, 12), dtype=bool)
    misaligned[:, :3] = True

    assert edge_alignment_score(image, aligned) > edge_alignment_score(image, misaligned)
