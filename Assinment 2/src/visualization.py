from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_annotation_overlay(
    image_rgb: np.ndarray, fg_seed: np.ndarray, bg_seed: np.ndarray
) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    overlay[fg_seed] = 0.55 * overlay[fg_seed] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
    overlay[bg_seed] = 0.55 * overlay[bg_seed] + 0.45 * np.array([0, 102, 255], dtype=np.float32)
    return overlay.astype(np.uint8)


def create_mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    color: tuple[int, int, int] = (0, 200, 83),
) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_arr
    return overlay.astype(np.uint8)


def save_comparison_panel(
    output_path: str | Path,
    image_rgb: np.ndarray,
    annotation_overlay: np.ndarray,
    baseline_mask: np.ndarray,
    raw_mask: np.ndarray,
    refined_mask: np.ndarray,
    final_overlay: np.ndarray,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=dpi)
    panels = [
        (image_rgb, "Original Image", None),
        (annotation_overlay, "User Annotations", None),
        (baseline_mask, "Naive Baseline", "gray"),
        (raw_mask, "Graph Cut (Raw)", "gray"),
        (refined_mask, "Graph Cut (Refined)", "gray"),
        (final_overlay, "Final Overlay", None),
    ]
    for axis, (panel, title, cmap) in zip(axes.ravel(), panels):
        axis.imshow(panel, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def save_boundary_comparison(
    output_path: str | Path,
    image_rgb: np.ndarray,
    raw_mask: np.ndarray,
    refined_mask: np.ndarray,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6), dpi=dpi)
    axis.imshow(image_rgb)
    axis.contour(raw_mask.astype(float), levels=[0.5], colors=["#ffb300"], linewidths=1.5)
    axis.contour(refined_mask.astype(float), levels=[0.5], colors=["#00c853"], linewidths=1.5)
    axis.set_title("Boundary Refinement: Raw vs Refined")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def save_iteration_plot(output_path: str | Path, energies: list[dict[str, float]], dpi: int) -> None:
    if not energies:
        return
    iterations = [entry["iteration"] for entry in energies]
    total = [entry["total_energy"] for entry in energies]
    data = [entry["data_energy"] for entry in energies]
    smooth = [entry["smooth_energy"] for entry in energies]

    figure, axis = plt.subplots(figsize=(7, 4), dpi=dpi)
    axis.plot(iterations, total, marker="o", label="Total Energy")
    axis.plot(iterations, data, marker="s", label="Data Energy")
    axis.plot(iterations, smooth, marker="^", label="Smoothness Energy")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Energy")
    axis.set_title("Graph-Cut Energy Across Iterations")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
