from __future__ import annotations

from pathlib import Path

from src.config import load_experiment_config
from src.data_io import load_case
from src.optimization import run_segmentation
from src.prepare_data import bundle_sample_dataset


def test_smoke_run_on_bundled_image() -> None:
    project_root = Path(__file__).resolve().parents[1]
    bundle_sample_dataset(project_root)
    config = load_experiment_config(project_root / "configs" / "experiment.yaml")
    item = config.dataset_items[0]
    case = load_case(item, max_dim=128)

    result = run_segmentation(case.image_rgb, case.fg_seed, case.bg_seed, config, bbox=case.bbox)

    assert result.baseline_mask.shape == case.image_rgb.shape[:2]
    assert result.raw_mask.shape == case.image_rgb.shape[:2]
    assert result.refined_mask.shape == case.image_rgb.shape[:2]
    assert len(result.iteration_records) >= 1
    assert result.iteration_records[-1].total_energy > 0.0
