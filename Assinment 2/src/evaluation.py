from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import perimeter

from src.graph_construction import PairwiseWeights


@dataclass(slots=True)
class IterationRecord:
    iteration: int
    data_energy: float
    smooth_energy: float
    total_energy: float
    maxflow_value: float
    mask_change: float
    foreground_fraction: float
    runtime_seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_data_energy(mask: np.ndarray, fg_cost: np.ndarray, bg_cost: np.ndarray) -> float:
    return float(np.sum(np.where(mask, fg_cost, bg_cost)))


def compute_smooth_energy(mask: np.ndarray, pairwise: PairwiseWeights) -> float:
    mask = mask.astype(bool)
    energy = 0.0
    energy += float(np.sum(pairwise.right[:, :-1] * (mask[:, :-1] != mask[:, 1:])))
    energy += float(np.sum(pairwise.down[:-1, :] * (mask[:-1, :] != mask[1:, :])))
    energy += float(
        np.sum(pairwise.down_right[:-1, :-1] * (mask[:-1, :-1] != mask[1:, 1:]))
    )
    energy += float(
        np.sum(pairwise.down_left[:-1, 1:] * (mask[:-1, 1:] != mask[1:, :-1]))
    )
    return energy


def count_connected_components(mask: np.ndarray) -> int:
    _, count = ndi.label(mask)
    return int(count)


def boundary_length(mask: np.ndarray) -> float:
    return float(perimeter(mask.astype(np.uint8), neighborhood=8))


def compactness(mask: np.ndarray) -> float:
    area = float(np.sum(mask))
    if area <= 0.0:
        return 0.0
    boundary = boundary_length(mask)
    if boundary <= 0.0:
        return 1.0
    return float((4.0 * np.pi * area) / (boundary * boundary))


def edge_alignment_score(image_rgb: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(grad_x, grad_y)

    eroded = ndi.binary_erosion(mask, iterations=1, border_value=0)
    dilated = ndi.binary_dilation(mask, iterations=1, border_value=0)
    boundary = dilated ^ eroded
    if not np.any(boundary):
        return 0.0

    boundary_strength = float(np.mean(gradient[boundary]))
    global_strength = float(np.mean(gradient)) + 1e-6
    return boundary_strength / global_strength


def bbox_leakage_ratio(mask: np.ndarray, bbox: tuple[int, int, int, int] | None) -> float:
    if bbox is None or not np.any(mask):
        return 0.0
    x1, y1, x2, y2 = bbox
    bbox_mask = np.zeros_like(mask, dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True
    outside = mask & ~bbox_mask
    return float(np.sum(outside) / max(np.sum(mask), 1))


def bbox_fill_ratio(mask: np.ndarray, bbox: tuple[int, int, int, int] | None) -> float:
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    bbox_area = max((y2 - y1) * (x2 - x1), 1)
    inside_foreground = np.sum(mask[y1:y2, x1:x2])
    return float(inside_foreground / bbox_area)


def seed_consistency_rate(mask: np.ndarray, fg_seed: np.ndarray, bg_seed: np.ndarray) -> float:
    correct_fg = float(np.mean(mask[fg_seed])) if np.any(fg_seed) else 1.0
    correct_bg = float(np.mean(~mask[bg_seed])) if np.any(bg_seed) else 1.0
    seed_count = int(np.sum(fg_seed) + np.sum(bg_seed))
    if seed_count == 0:
        return 1.0
    return ((correct_fg * int(np.sum(fg_seed))) + (correct_bg * int(np.sum(bg_seed)))) / seed_count


def mask_change_fraction(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return 1.0
    return float(np.mean(previous != current))


def summarize_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    fg_cost: np.ndarray,
    bg_cost: np.ndarray,
    pairwise: PairwiseWeights,
    fg_seed: np.ndarray,
    bg_seed: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
) -> dict[str, float]:
    data_energy = compute_data_energy(mask, fg_cost, bg_cost)
    smooth_energy = compute_smooth_energy(mask, pairwise)
    return {
        "data_energy": data_energy,
        "smooth_energy": smooth_energy,
        "total_energy": data_energy + smooth_energy,
        "foreground_fraction": float(np.mean(mask)),
        "component_count": float(count_connected_components(mask)),
        "boundary_length": boundary_length(mask),
        "compactness": compactness(mask),
        "edge_alignment_score": edge_alignment_score(image_rgb, mask),
        "bbox_leakage_ratio": bbox_leakage_ratio(mask, bbox),
        "bbox_fill_ratio": bbox_fill_ratio(mask, bbox),
        "seed_consistency_rate": seed_consistency_rate(mask, fg_seed, bg_seed),
    }


def aggregate_case_summaries(rows: list[dict[str, float | str]]) -> dict[str, float | int]:
    numeric_keys = [
        key for key, value in rows[0].items() if isinstance(value, (int, float))
    ]
    summary: dict[str, float | int] = {"case_count": len(rows)}
    for key in numeric_keys:
        summary[f"mean_{key}"] = float(np.mean([float(row[key]) for row in rows]))
    return summary
