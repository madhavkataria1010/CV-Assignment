from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetItem:
    name: str
    image_path: Path
    fg_scribble_path: Path | None
    bg_scribble_path: Path | None
    bbox: tuple[int, int, int, int] | None = None
    target_label: str = ""
    description: str = ""
    report_caption: str = ""


@dataclass(slots=True)
class RefinementConfig:
    min_object_size: int = 96
    min_hole_size: int = 96
    opening_radius: int = 1
    closing_radius: int = 3
    smoothing_sigma: float = 1.0


@dataclass(slots=True)
class VisualizationConfig:
    overlay_alpha: float = 0.45
    figure_dpi: int = 180


@dataclass(slots=True)
class ExperimentConfig:
    dataset_config: Path
    output_dir: Path
    report_figure_dir: Path
    max_dim: int
    histogram_bins: tuple[int, int, int]
    lambda_smooth: float
    hard_seed_cost: float
    bbox_outside_penalty: float
    distance_prior_weight: float
    probability_smoothing: float
    max_iterations: int
    convergence_tol: float
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    dataset_items: list[DatasetItem] = field(default_factory=list)


def _resolve_path_flexible(config_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    candidates = [
        (config_path.parent / raw_path).resolve(),
        (config_path.parent.parent / raw_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_output_path(config_path: Path, value: str | None, default: str) -> Path:
    raw_path = Path(value or default)
    if raw_path.is_absolute():
        return raw_path
    return (config_path.parent.parent / raw_path).resolve()


def _to_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return (
            int(value["x"]),
            int(value["y"]),
            int(value["width"]),
            int(value["height"]),
        )
    if len(value) != 4:
        raise ValueError(f"Expected bbox with 4 values, received: {value}")
    return tuple(int(v) for v in value)


def load_dataset_config(dataset_path: str | Path) -> list[DatasetItem]:
    dataset_path = Path(dataset_path).resolve()
    payload = yaml.safe_load(dataset_path.read_text()) or {}
    items = []
    for entry in payload.get("items", payload.get("images", [])):
        items.append(
            DatasetItem(
                name=str(entry["name"]),
                image_path=_resolve_path_flexible(dataset_path, entry["image_path"]),
                fg_scribble_path=_resolve_path_flexible(dataset_path, entry.get("fg_scribble_path")),
                bg_scribble_path=_resolve_path_flexible(dataset_path, entry.get("bg_scribble_path")),
                bbox=_to_bbox(entry.get("bbox")),
                target_label=str(entry.get("target_label", "")),
                description=str(entry.get("description", "")),
                report_caption=str(entry.get("report_caption", "")),
            )
        )
    if not items:
        raise ValueError(f"No dataset items found in {dataset_path}")
    return items


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path).resolve()
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_config = _resolve_path_flexible(config_path, payload["dataset_config"])
    refinement_payload = payload.get("refinement") or {}
    visualization_payload = payload.get("visualization") or {}
    histogram_bins = payload.get("histogram_bins")
    color_bins = int(payload.get("color_bins", histogram_bins[0] if histogram_bins else 16))
    experiment = ExperimentConfig(
        dataset_config=dataset_config,
        output_dir=_resolve_output_path(config_path, payload.get("output_dir"), "results"),
        report_figure_dir=_resolve_output_path(
            config_path,
            payload.get("report_figure_dir"),
            "report/figures",
        ),
        max_dim=int(payload.get("max_dim", 512)),
        histogram_bins=(color_bins, color_bins, color_bins),
        lambda_smooth=float(payload.get("lambda_smooth", 28.0)),
        hard_seed_cost=float(payload.get("hard_seed_cost", payload.get("hard_constraint_cost", 1e6))),
        bbox_outside_penalty=float(payload.get("bbox_outside_penalty", 4.0)),
        distance_prior_weight=float(payload.get("distance_prior_weight", 2.0)),
        probability_smoothing=float(payload.get("probability_smoothing", 1e-3)),
        max_iterations=int(payload.get("max_iterations", 4)),
        convergence_tol=float(payload.get("convergence_tol", 0.002)),
        refinement=RefinementConfig(
            min_object_size=int(payload.get("min_object_size", refinement_payload.get("min_object_size", 96))),
            min_hole_size=int(payload.get("min_hole_size", refinement_payload.get("min_hole_size", 96))),
            opening_radius=int(payload.get("opening_radius", refinement_payload.get("opening_radius", 1))),
            closing_radius=int(payload.get("closing_radius", refinement_payload.get("closing_radius", payload.get("morph_radius", 3)))),
            smoothing_sigma=float(payload.get("smoothing_sigma", refinement_payload.get("smoothing_sigma", 1.0))),
        ),
        visualization=VisualizationConfig(
            overlay_alpha=float(payload.get("overlay_alpha", visualization_payload.get("overlay_alpha", 0.45))),
            figure_dpi=int(payload.get("figure_dpi", visualization_payload.get("figure_dpi", 180))),
        ),
    )
    experiment.dataset_items = load_dataset_config(dataset_config)
    return experiment
