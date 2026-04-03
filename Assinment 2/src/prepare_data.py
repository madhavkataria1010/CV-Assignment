from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from skimage import data

from src.data_io import ensure_dir


def _draw_scribble(mask: np.ndarray, points: list[tuple[int, int]], thickness: int) -> None:
    for start, end in zip(points[:-1], points[1:]):
        cv2.line(mask, start, end, color=255, thickness=thickness)


def _scaled_points(width: int, height: int, points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(int(round(width * x)), int(round(height * y))) for x, y in points]


def _case_specs() -> dict[str, dict[str, object]]:
    return {
        "astronaut": {
            "target_label": "astronaut subject",
            "bbox": (110, 25, 405, 510),
            "fg_paths": [
                [(0.42, 0.17), (0.47, 0.29), (0.52, 0.42)],
                [(0.38, 0.44), (0.34, 0.62), (0.32, 0.80)],
                [(0.58, 0.46), (0.62, 0.60), (0.66, 0.74)],
            ],
            "bg_paths": [
                [(0.05, 0.08), (0.14, 0.10)],
                [(0.80, 0.10), (0.91, 0.14)],
                [(0.08, 0.82), (0.20, 0.90)],
                [(0.83, 0.82), (0.92, 0.94)],
                [(0.79, 0.38), (0.86, 0.52)],
            ],
        },
        "coffee": {
            "target_label": "coffee cup",
            "bbox": (180, 85, 428, 340),
            "fg_paths": [
                [(0.44, 0.24), (0.52, 0.30), (0.54, 0.42)],
                [(0.39, 0.48), (0.42, 0.63), (0.53, 0.67)],
                [(0.56, 0.52), (0.60, 0.56), (0.62, 0.50)],
            ],
            "bg_paths": [
                [(0.06, 0.10), (0.18, 0.16)],
                [(0.78, 0.14), (0.92, 0.22)],
                [(0.10, 0.84), (0.24, 0.90)],
                [(0.74, 0.82), (0.90, 0.88)],
            ],
        },
        "chelsea": {
            "target_label": "cat",
            "bbox": (55, 15, 350, 295),
            "fg_paths": [
                [(0.25, 0.28), (0.36, 0.34), (0.48, 0.40)],
                [(0.53, 0.34), (0.50, 0.48), (0.44, 0.62)],
                [(0.30, 0.48), (0.40, 0.56), (0.54, 0.58)],
            ],
            "bg_paths": [
                [(0.05, 0.08), (0.12, 0.18)],
                [(0.82, 0.14), (0.93, 0.22)],
                [(0.08, 0.82), (0.20, 0.90)],
                [(0.76, 0.78), (0.92, 0.90)],
            ],
        },
    }


def bundle_sample_dataset(project_root: str | Path) -> None:
    project_root = Path(project_root)
    input_dir = ensure_dir(project_root / "data" / "input")
    annotation_dir = ensure_dir(project_root / "data" / "annotations")

    images = {
        "astronaut": data.astronaut(),
        "coffee": data.coffee(),
        "chelsea": data.chelsea(),
    }
    specs = _case_specs()

    for name, image in images.items():
        image = image.astype(np.uint8)
        height, width = image.shape[:2]
        fg_mask = np.zeros((height, width), dtype=np.uint8)
        bg_mask = np.zeros((height, width), dtype=np.uint8)
        thickness = max(6, min(height, width) // 28)

        for fg_path in specs[name]["fg_paths"]:
            _draw_scribble(fg_mask, _scaled_points(width, height, fg_path), thickness)
        for bg_path in specs[name]["bg_paths"]:
            _draw_scribble(bg_mask, _scaled_points(width, height, bg_path), thickness)

        iio.imwrite(input_dir / f"{name}.png", image)
        iio.imwrite(annotation_dir / f"{name}_fg.png", fg_mask)
        iio.imwrite(annotation_dir / f"{name}_bg.png", bg_mask)


if __name__ == "__main__":
    bundle_sample_dataset(Path(__file__).resolve().parent.parent)
