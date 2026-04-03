from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class HistogramModel:
    probabilities: np.ndarray
    bins: tuple[int, int, int]


def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _bin_indices(image_lab: np.ndarray, bins: tuple[int, int, int]) -> np.ndarray:
    bins_arr = np.asarray(bins, dtype=np.float32)
    clipped = np.clip(image_lab, 0.0, 255.0)
    scaled = np.floor(clipped * (bins_arr / 256.0)).astype(np.int32)
    max_vals = np.asarray(bins, dtype=np.int32) - 1
    return np.minimum(scaled, max_vals)


def fit_histogram_model(
    image_lab: np.ndarray,
    mask: np.ndarray,
    bins: tuple[int, int, int],
    smoothing: float,
) -> HistogramModel:
    pixels = image_lab[mask]
    if pixels.size == 0:
        raise ValueError("Cannot fit histogram model on an empty mask.")
    histogram, _ = np.histogramdd(
        pixels,
        bins=bins,
        range=((0, 256), (0, 256), (0, 256)),
    )
    histogram = histogram.astype(np.float64) + float(smoothing)
    probabilities = histogram / np.sum(histogram)
    return HistogramModel(probabilities=probabilities, bins=bins)


def negative_log_likelihood(image_lab: np.ndarray, model: HistogramModel) -> np.ndarray:
    indices = _bin_indices(image_lab, model.bins)
    probs = model.probabilities[
        indices[..., 0],
        indices[..., 1],
        indices[..., 2],
    ]
    return -np.log(probs + 1e-12).astype(np.float32)


def compute_unary_costs(
    image_lab: np.ndarray,
    fg_model: HistogramModel,
    bg_model: HistogramModel,
    fg_seed: np.ndarray,
    bg_seed: np.ndarray,
    hard_seed_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    fg_cost = negative_log_likelihood(image_lab, fg_model)
    bg_cost = negative_log_likelihood(image_lab, bg_model)
    fg_cost = fg_cost.astype(np.float32)
    bg_cost = bg_cost.astype(np.float32)
    fg_cost[fg_seed] = 0.0
    bg_cost[fg_seed] = float(hard_seed_cost)
    fg_cost[bg_seed] = float(hard_seed_cost)
    bg_cost[bg_seed] = 0.0
    return fg_cost, bg_cost


def initialize_models(
    image_lab: np.ndarray,
    fg_seed: np.ndarray,
    bg_seed: np.ndarray,
    bins: tuple[int, int, int],
    smoothing: float,
) -> tuple[HistogramModel, HistogramModel]:
    fg_model = fit_histogram_model(image_lab, fg_seed, bins, smoothing)
    bg_model = fit_histogram_model(image_lab, bg_seed, bins, smoothing)
    return fg_model, bg_model


def update_models_from_mask(
    image_lab: np.ndarray,
    mask: np.ndarray,
    fg_seed: np.ndarray,
    bg_seed: np.ndarray,
    bins: tuple[int, int, int],
    smoothing: float,
) -> tuple[HistogramModel, HistogramModel]:
    fg_mask = (mask | fg_seed) & ~bg_seed
    bg_mask = ((~mask) | bg_seed) & ~fg_seed
    return initialize_models(image_lab, fg_mask, bg_mask, bins, smoothing)
