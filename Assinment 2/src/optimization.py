from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from src.baseline import run_naive_segmentation
from src.config import ExperimentConfig
from src.evaluation import (
    IterationRecord,
    compute_data_energy,
    compute_smooth_energy,
    mask_change_fraction,
)
from src.graph_construction import PairwiseWeights, compute_pairwise_weights
from src.maxflow_solver import solve_graph_cut, terminal_capacities_from_costs
from src.modeling import (
    compute_unary_costs,
    initialize_models,
    rgb_to_lab,
    update_models_from_mask,
)
from src.refinement import refine_mask


@dataclass(slots=True)
class SegmentationResult:
    baseline_mask: np.ndarray
    raw_mask: np.ndarray
    refined_mask: np.ndarray
    pairwise: PairwiseWeights
    final_fg_cost: np.ndarray
    final_bg_cost: np.ndarray
    baseline_fg_cost: np.ndarray
    baseline_bg_cost: np.ndarray
    iteration_records: list[IterationRecord]
    beta: float


def run_segmentation(
    image_rgb: np.ndarray,
    fg_seed: np.ndarray,
    bg_seed: np.ndarray,
    config: ExperimentConfig,
    bbox: tuple[int, int, int, int] | None = None,
) -> SegmentationResult:
    image_lab = rgb_to_lab(image_rgb)
    fg_model, bg_model = initialize_models(
        image_lab,
        fg_seed,
        bg_seed,
        config.histogram_bins,
        config.probability_smoothing,
    )

    baseline_fg_cost, baseline_bg_cost = compute_unary_costs(
        image_lab,
        fg_model,
        bg_model,
        fg_seed,
        bg_seed,
        config.hard_seed_cost,
    )
    distance_fg_cost, distance_bg_cost = compute_distance_prior_costs(
        fg_seed,
        bg_seed,
        config.distance_prior_weight,
    )
    baseline_fg_cost = baseline_fg_cost + distance_fg_cost
    baseline_bg_cost = baseline_bg_cost + distance_bg_cost
    baseline_fg_cost, baseline_bg_cost = apply_bbox_prior(
        baseline_fg_cost,
        baseline_bg_cost,
        bbox=bbox,
        penalty=config.bbox_outside_penalty,
    )
    baseline_mask = run_naive_segmentation(baseline_fg_cost, baseline_bg_cost, fg_seed, bg_seed)
    pairwise = compute_pairwise_weights(image_lab, config.lambda_smooth)

    iteration_records: list[IterationRecord] = []
    previous_mask: np.ndarray | None = None
    current_mask = baseline_mask.copy()
    final_fg_cost = baseline_fg_cost
    final_bg_cost = baseline_bg_cost

    for iteration in range(config.max_iterations):
        fg_cost, bg_cost = compute_unary_costs(
            image_lab,
            fg_model,
            bg_model,
            fg_seed,
            bg_seed,
            config.hard_seed_cost,
        )
        fg_cost = fg_cost + distance_fg_cost
        bg_cost = bg_cost + distance_bg_cost
        fg_cost, bg_cost = apply_bbox_prior(
            fg_cost, bg_cost, bbox=bbox, penalty=config.bbox_outside_penalty
        )
        source_caps, sink_caps = terminal_capacities_from_costs(fg_cost, bg_cost)
        start = time.perf_counter()
        solver_result = solve_graph_cut(source_caps, sink_caps, pairwise)
        runtime = time.perf_counter() - start
        current_mask = solver_result.mask

        data_energy = compute_data_energy(current_mask, fg_cost, bg_cost)
        smooth_energy = compute_smooth_energy(current_mask, pairwise)
        change = mask_change_fraction(previous_mask, current_mask)
        iteration_records.append(
            IterationRecord(
                iteration=iteration + 1,
                data_energy=data_energy,
                smooth_energy=smooth_energy,
                total_energy=data_energy + smooth_energy,
                maxflow_value=solver_result.maxflow_value,
                mask_change=change,
                foreground_fraction=float(np.mean(current_mask)),
                runtime_seconds=runtime,
            )
        )

        final_fg_cost = fg_cost
        final_bg_cost = bg_cost

        if previous_mask is not None and change <= config.convergence_tol:
            break

        previous_mask = current_mask.copy()
        fg_model, bg_model = update_models_from_mask(
            image_lab,
            current_mask,
            fg_seed,
            bg_seed,
            config.histogram_bins,
            config.probability_smoothing,
        )

    refined = refine_mask(current_mask, fg_seed, bg_seed, config.refinement)
    return SegmentationResult(
        baseline_mask=baseline_mask,
        raw_mask=current_mask,
        refined_mask=refined,
        pairwise=pairwise,
        final_fg_cost=final_fg_cost,
        final_bg_cost=final_bg_cost,
        baseline_fg_cost=baseline_fg_cost,
        baseline_bg_cost=baseline_bg_cost,
        iteration_records=iteration_records,
        beta=pairwise.beta,
    )


def compute_distance_prior_costs(
    fg_seed: np.ndarray,
    bg_seed: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    shape = fg_seed.shape
    zeros = np.zeros(shape, dtype=np.float32)
    if weight <= 0:
        return zeros, zeros.copy()

    fg_distance = ndi.distance_transform_edt(~fg_seed).astype(np.float32)
    bg_distance = ndi.distance_transform_edt(~bg_seed).astype(np.float32)
    diagonal = max(float(np.hypot(*shape)), 1.0)

    fg_cost = weight * np.log1p(fg_distance) / np.log1p(diagonal)
    bg_cost = weight * np.log1p(bg_distance) / np.log1p(diagonal)
    return fg_cost.astype(np.float32), bg_cost.astype(np.float32)


def apply_bbox_prior(
    fg_cost: np.ndarray,
    bg_cost: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    if bbox is None or penalty <= 0:
        return fg_cost, bg_cost
    height, width = fg_cost.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - 4)
    y1 = max(0, y1 - 4)
    x2 = min(width, x2 + 4)
    y2 = min(height, y2 + 4)
    fg_cost = fg_cost.copy()
    outside = np.ones_like(fg_cost, dtype=bool)
    outside[y1:y2, x1:x2] = False
    fg_cost[outside] += float(penalty)
    return fg_cost, bg_cost
