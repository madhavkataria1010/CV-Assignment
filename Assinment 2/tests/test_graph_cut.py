from __future__ import annotations

import numpy as np

from src.graph_construction import PairwiseWeights
from src.maxflow_solver import solve_graph_cut


def test_graph_cut_reduces_to_unary_solution_when_pairwise_zero() -> None:
    source_caps = np.array([[10.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    sink_caps = np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float32)
    zeros = np.zeros_like(source_caps, dtype=np.float32)
    pairwise = PairwiseWeights(
        right=zeros.copy(),
        down=zeros.copy(),
        down_right=zeros.copy(),
        down_left=zeros.copy(),
        beta=1.0,
    )

    result = solve_graph_cut(source_caps, sink_caps, pairwise)
    expected = np.array([[True, False], [True, False]])

    assert np.array_equal(result.mask, expected)
    assert result.maxflow_value >= 0.0
