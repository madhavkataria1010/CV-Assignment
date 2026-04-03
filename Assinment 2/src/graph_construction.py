from __future__ import annotations

from dataclasses import dataclass

import maxflow
import numpy as np


RIGHT_STRUCTURE = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
DOWN_STRUCTURE = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=np.int32)
DOWN_RIGHT_STRUCTURE = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.int32)
DOWN_LEFT_STRUCTURE = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int32)


@dataclass(slots=True)
class PairwiseWeights:
    right: np.ndarray
    down: np.ndarray
    down_right: np.ndarray
    down_left: np.ndarray
    beta: float


def _valid_mean(values: list[np.ndarray]) -> float:
    flattened = [value.reshape(-1) for value in values if value.size > 0]
    if not flattened:
        return 1.0
    concatenated = np.concatenate(flattened)
    return float(np.mean(concatenated))


def compute_pairwise_weights(image_lab: np.ndarray, lambda_smooth: float) -> PairwiseWeights:
    image_lab = image_lab.astype(np.float32)
    right_diff = np.sum((image_lab[:, :-1] - image_lab[:, 1:]) ** 2, axis=2)
    down_diff = np.sum((image_lab[:-1, :] - image_lab[1:, :]) ** 2, axis=2)
    down_right_diff = np.sum((image_lab[:-1, :-1] - image_lab[1:, 1:]) ** 2, axis=2)
    down_left_diff = np.sum((image_lab[:-1, 1:] - image_lab[1:, :-1]) ** 2, axis=2)

    beta_denom = _valid_mean([right_diff, down_diff, down_right_diff, down_left_diff])
    beta = 1.0 / max(2.0 * beta_denom, 1e-12)

    height, width = image_lab.shape[:2]
    right = np.zeros((height, width), dtype=np.float32)
    down = np.zeros((height, width), dtype=np.float32)
    down_right = np.zeros((height, width), dtype=np.float32)
    down_left = np.zeros((height, width), dtype=np.float32)

    right[:, :-1] = lambda_smooth * np.exp(-beta * right_diff)
    down[:-1, :] = lambda_smooth * np.exp(-beta * down_diff)
    down_right[:-1, :-1] = (lambda_smooth / np.sqrt(2.0)) * np.exp(-beta * down_right_diff)
    down_left[:-1, 1:] = (lambda_smooth / np.sqrt(2.0)) * np.exp(-beta * down_left_diff)

    return PairwiseWeights(
        right=right,
        down=down,
        down_right=down_right,
        down_left=down_left,
        beta=beta,
    )


def build_graph(
    source_caps: np.ndarray,
    sink_caps: np.ndarray,
    pairwise: PairwiseWeights,
) -> tuple[maxflow.GraphFloat, np.ndarray]:
    graph = maxflow.Graph[float]()
    nodeids = graph.add_grid_nodes(source_caps.shape)
    graph.add_grid_tedges(nodeids, source_caps.astype(np.float64), sink_caps.astype(np.float64))
    graph.add_grid_edges(nodeids, pairwise.right, RIGHT_STRUCTURE, symmetric=True)
    graph.add_grid_edges(nodeids, pairwise.down, DOWN_STRUCTURE, symmetric=True)
    graph.add_grid_edges(nodeids, pairwise.down_right, DOWN_RIGHT_STRUCTURE, symmetric=True)
    graph.add_grid_edges(nodeids, pairwise.down_left, DOWN_LEFT_STRUCTURE, symmetric=True)
    return graph, nodeids
