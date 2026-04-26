from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch

TARGET_SIZE = 28
INNER_SIZE = 20
BACKGROUND_BLUR_RADIUS = 12
_RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class PreprocessingStages:
    original: np.ndarray
    contrast: np.ndarray
    thresholded: np.ndarray
    cropped: np.ndarray
    resized: np.ndarray
    centered: np.ndarray
    final_28x28: np.ndarray
    bbox: tuple[int, int, int, int]
    threshold: int


def load_grayscale_image(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def enhance_foreground(img: np.ndarray, blur_radius: int = BACKGROUND_BLUR_RADIUS) -> np.ndarray:
    pil_img = Image.fromarray(img)
    background = np.array(
        pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
        dtype=np.int16,
    )
    foreground = np.clip(background - img.astype(np.int16), 0, 255)
    return foreground.astype(np.uint8)


def otsu_threshold_value(img: np.ndarray) -> int:
    histogram = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = img.size
    weighted_sum = np.dot(np.arange(256), histogram)

    background_sum = 0.0
    background_weight = 0.0
    best_threshold = 0
    best_between_class_variance = -1.0

    for threshold in range(256):
        background_weight += histogram[threshold]
        if background_weight == 0:
            continue

        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break

        background_sum += threshold * histogram[threshold]
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_sum - background_sum) / foreground_weight

        between_class_variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if between_class_variance > best_between_class_variance:
            best_between_class_variance = between_class_variance
            best_threshold = threshold

    return int(best_threshold)


def threshold_image(img: np.ndarray, threshold: int | None = None) -> tuple[np.ndarray, int]:
    threshold_value = otsu_threshold_value(img) if threshold is None else int(threshold)
    binary = (img >= threshold_value).astype(np.uint8)
    return binary, threshold_value


def find_bounding_box(binary: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(binary > 0)
    if len(coords) == 0:
        raise ValueError("No foreground pixels found.")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(y_min), int(y_max), int(x_min), int(x_max)


def crop_to_content(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y_min, y_max, x_min, x_max = bbox
    return img[y_min : y_max + 1, x_min : x_max + 1]


def resize_preserve_aspect(img: np.ndarray, target_inner: int = INNER_SIZE) -> np.ndarray:
    height, width = img.shape
    scale = target_inner / max(height, width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    pil_img = Image.fromarray(img)
    resized = pil_img.resize((new_width, new_height), _RESAMPLING.LANCZOS)
    return np.array(resized, dtype=np.uint8)


def _center_of_mass(img: np.ndarray) -> tuple[float, float]:
    weights = img.astype(np.float64)
    total_weight = weights.sum()
    if total_weight == 0:
        y_midpoint = (img.shape[0] - 1) / 2.0
        x_midpoint = (img.shape[1] - 1) / 2.0
        return y_midpoint, x_midpoint

    ys, xs = np.indices(img.shape)
    y_center = float((ys * weights).sum() / total_weight)
    x_center = float((xs * weights).sum() / total_weight)
    return y_center, x_center


def _shift_image(img: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    height, width = img.shape
    shifted = np.zeros_like(img)

    src_y_min = max(0, -shift_y)
    src_y_max = min(height, height - shift_y)
    src_x_min = max(0, -shift_x)
    src_x_max = min(width, width - shift_x)

    dst_y_min = max(0, shift_y)
    dst_x_min = max(0, shift_x)
    dst_y_max = dst_y_min + (src_y_max - src_y_min)
    dst_x_max = dst_x_min + (src_x_max - src_x_min)

    if src_y_max > src_y_min and src_x_max > src_x_min:
        shifted[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = img[
            src_y_min:src_y_max,
            src_x_min:src_x_max,
        ]

    return shifted


def center_on_canvas(img: np.ndarray, canvas_size: int = TARGET_SIZE) -> np.ndarray:
    height, width = img.shape
    if height > canvas_size or width > canvas_size:
        raise ValueError("Image is larger than the target canvas.")

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y_offset = (canvas_size - height) // 2
    x_offset = (canvas_size - width) // 2
    canvas[y_offset : y_offset + height, x_offset : x_offset + width] = img

    y_center, x_center = _center_of_mass(canvas)
    target_center = (canvas_size - 1) / 2.0
    shift_y = int(round(target_center - y_center))
    shift_x = int(round(target_center - x_center))
    return _shift_image(canvas, shift_y, shift_x)


def normalize_for_model(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).to(dtype=torch.float32).div(255.0).unsqueeze(0)


def build_preprocessing_stages(path: str | Path) -> PreprocessingStages:
    gray = load_grayscale_image(path)
    contrast = enhance_foreground(gray)
    binary, threshold = threshold_image(contrast)
    bbox = find_bounding_box(binary)
    cropped = crop_to_content(binary * 255, bbox)
    resized = resize_preserve_aspect(cropped)
    centered = center_on_canvas(resized)

    return PreprocessingStages(
        original=gray,
        contrast=contrast,
        thresholded=binary * 255,
        cropped=cropped,
        resized=resized,
        centered=centered,
        final_28x28=centered,
        bbox=bbox,
        threshold=threshold,
    )


def preprocess_for_inference(path: str | Path) -> torch.Tensor:
    stages = build_preprocessing_stages(path)
    tensor = normalize_for_model(stages.final_28x28)
    return tensor.unsqueeze(0)
