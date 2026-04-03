from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.graph_construction import PairwiseWeights, build_graph


@dataclass(slots=True)
class SolverResult:
    mask: np.ndarray
    maxflow_value: float


def terminal_capacities_from_costs(
    fg_cost: np.ndarray, bg_cost: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source_caps = bg_cost.astype(np.float32)
    sink_caps = fg_cost.astype(np.float32)
    return source_caps, sink_caps


def solve_graph_cut(
    source_caps: np.ndarray,
    sink_caps: np.ndarray,
    pairwise: PairwiseWeights,
) -> SolverResult:
    graph, nodeids = build_graph(source_caps, sink_caps, pairwise)
    maxflow_value = float(graph.maxflow())
    sink_segments = graph.get_grid_segments(nodeids)
    mask = np.logical_not(sink_segments)
    return SolverResult(mask=mask.astype(bool), maxflow_value=maxflow_value)
