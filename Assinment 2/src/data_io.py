from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np

from src.config import DatasetItem


@dataclass(slots=True)
class LoadedCase:
    name: str
    image_rgb: np.ndarray
    fg_seed: np.ndarray
    bg_seed: np.ndarray
    bbox: tuple[int, int, int, int] | None
    target_label: str
    original_shape: tuple[int, int, int]
    resized_shape: tuple[int, int, int]
    scale_factor: float


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_rgb_image(path: str | Path) -> np.ndarray:
    image = iio.imread(path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.uint8)


def load_binary_mask(path: str | Path) -> np.ndarray:
    mask = iio.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8) > 0


def resize_image(image_rgb: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    if max_dim <= 0:
        return image_rgb, 1.0
    height, width = image_rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image_rgb, 1.0
    scale = max_dim / float(longest)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    resized = cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def create_bbox_seed_masks(
    shape: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    x1, y1, x2, y2 = bbox
    fg_seed = np.zeros((height, width), dtype=bool)
    bg_seed = np.zeros((height, width), dtype=bool)
    fg_inset_x = max(1, (x2 - x1) // 5)
    fg_inset_y = max(1, (y2 - y1) // 5)
    fg_seed[y1 + fg_inset_y : y2 - fg_inset_y, x1 + fg_inset_x : x2 - fg_inset_x] = True
    bg_seed[: max(1, y1 // 2), :] = True
    bg_seed[min(height - 1, y2 + max(1, (height - y2) // 3)) :, :] = True
    bg_seed[:, : max(1, x1 // 2)] = True
    bg_seed[:, min(width - 1, x2 + max(1, (width - x2) // 3)) :] = True
    bg_seed[y1:y2, x1:x2] = False
    return fg_seed, bg_seed


def load_case(item: DatasetItem, max_dim: int) -> LoadedCase:
    image_rgb = load_rgb_image(item.image_path)
    resized_image, scale = resize_image(image_rgb, max_dim)
    target_size = (resized_image.shape[1], resized_image.shape[0])

    if item.fg_scribble_path and item.bg_scribble_path:
        fg_seed = resize_mask(load_binary_mask(item.fg_scribble_path), target_size)
        bg_seed = resize_mask(load_binary_mask(item.bg_scribble_path), target_size)
    elif item.bbox is not None:
        bbox = scale_bbox(item.bbox, scale)
        fg_seed, bg_seed = create_bbox_seed_masks(resized_image.shape[:2], bbox)
    else:
        raise ValueError(f"Case {item.name} is missing both scribbles and bbox guidance.")

    fg_seed, bg_seed = validate_seed_masks(fg_seed, bg_seed)
    bbox = scale_bbox(item.bbox, scale) if item.bbox else None
    return LoadedCase(
        name=item.name,
        image_rgb=resized_image,
        fg_seed=fg_seed,
        bg_seed=bg_seed,
        bbox=bbox,
        target_label=item.target_label,
        original_shape=image_rgb.shape,
        resized_shape=resized_image.shape,
        scale_factor=scale,
    )


def scale_bbox(bbox: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    return tuple(int(round(value * scale)) for value in bbox)


def validate_seed_masks(fg_seed: np.ndarray, bg_seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fg_seed = fg_seed.astype(bool)
    bg_seed = bg_seed.astype(bool)
    overlap = fg_seed & bg_seed
    if np.any(overlap):
        bg_seed = bg_seed & ~overlap
    if not np.any(fg_seed):
        raise ValueError("Foreground seed mask is empty.")
    if not np.any(bg_seed):
        raise ValueError("Background seed mask is empty.")
    return fg_seed, bg_seed


def case_output_dir(root: str | Path, case_name: str) -> Path:
    return ensure_dir(Path(root) / case_name)


def save_rgb_image(path: str | Path, image_rgb: np.ndarray) -> None:
    iio.imwrite(path, image_rgb.astype(np.uint8))


def save_binary_mask(path: str | Path, mask: np.ndarray) -> None:
    iio.imwrite(path, (mask.astype(np.uint8) * 255))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
