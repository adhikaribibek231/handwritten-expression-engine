"""Image preprocessing helpers for handwritten digit inference.

The job of this module is to take a raw photo or scan and slowly turn it into
something that looks close to MNIST:
load it, make the stroke stand out, threshold it, crop the digit, resize it,
center it on a 28x28 canvas, then convert it to a tensor for the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch
import cv2
TARGET_SIZE = 28
INNER_SIZE = 20
BACKGROUND_BLUR_RADIUS = 12
_RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class PreprocessingStages:
    """Snapshots from each preprocessing step so the pipeline is easy to inspect."""

    original: np.ndarray
    contrast: np.ndarray
    thresholded: np.ndarray
    cropped: np.ndarray
    resized: np.ndarray
    centered: np.ndarray
    final_28x28: np.ndarray
    bbox: tuple[int, int, int, int]
    threshold: int


# step 1 - load the raw image
def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Open an image file and return a grayscale numpy array."""

    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


# step 2 - make the stroke stand out from the page
def enhance_foreground(img: np.ndarray, blur_radius: int = BACKGROUND_BLUR_RADIUS) -> np.ndarray:
    """Make the dark stroke pop by subtracting a blurred background estimate."""

    pil_img = Image.fromarray(img)
    background = np.array(
        pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
        dtype=np.int16,
    )
    foreground = np.clip(background - img.astype(np.int16), 0, 255)
    return foreground.astype(np.uint8)


# step 3 - choose a threshold automatically
def otsu_threshold_value(img: np.ndarray) -> int:
    """Pick the cutoff that best separates foreground pixels from background pixels."""

    histogram = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = img.size
    weighted_sum = np.dot(np.arange(256), histogram)

    background_sum = 0.0
    background_weight = 0.0
    best_threshold = 0
    best_between_class_variance = -1.0

    # Try every possible gray level and keep the split with the cleanest separation.
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


# step 4 - turn grayscale into a binary mask
def threshold_image(img: np.ndarray, threshold: int | None = None) -> tuple[np.ndarray, int]:
    """Convert the contrast image into a 0/1 mask and report the threshold used."""

    threshold_value = otsu_threshold_value(img) if threshold is None else int(threshold)
    binary = (img >= threshold_value).astype(np.uint8)
    return binary, threshold_value


# step 5 - find the tight box around the digit
def find_bounding_box(binary: np.ndarray) -> tuple[int, int, int, int]:
    """Find the smallest box that still contains every foreground pixel."""

    coords = np.argwhere(binary > 0)
    if len(coords) == 0:
        raise ValueError("No foreground pixels found.")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(y_min), int(y_max), int(x_min), int(x_max)


# step 6 - crop away empty border
def crop_to_content(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop an image down to the content box returned by `find_bounding_box`."""

    y_min, y_max, x_min, x_max = bbox
    return img[y_min : y_max + 1, x_min : x_max + 1]


# step 7 - resize while keeping the digit shape
def resize_preserve_aspect(img: np.ndarray, target_inner: int = INNER_SIZE) -> np.ndarray:
    """Resize the digit so its longest side fits the inner MNIST-like area."""

    height, width = img.shape
    scale = target_inner / max(height, width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    pil_img = Image.fromarray(img)
    resized = pil_img.resize((new_width, new_height), _RESAMPLING.LANCZOS)
    return np.array(resized, dtype=np.uint8)


# step 8 - helper to measure where the pixel mass currently sits
def _center_of_mass(img: np.ndarray) -> tuple[float, float]:
    """Measure where the pixel mass is concentrated inside an image."""

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


# step 9 - helper to shift the digit without wrapping pixels around
def _shift_image(img: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    """Move an image on the canvas without wrapping pixels around the edges."""

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


# step 10 - place the digit on a 28x28 canvas and center it
def center_on_canvas(img: np.ndarray, canvas_size: int = TARGET_SIZE) -> np.ndarray:
    """Place the resized digit on a square canvas and nudge it toward the center."""

    height, width = img.shape
    if height > canvas_size or width > canvas_size:
        raise ValueError("Image is larger than the target canvas.")

    # First center by geometry, then do a tiny correction using the actual stroke mass.
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y_offset = (canvas_size - height) // 2
    x_offset = (canvas_size - width) // 2
    canvas[y_offset : y_offset + height, x_offset : x_offset + width] = img

    y_center, x_center = _center_of_mass(canvas)
    target_center = (canvas_size - 1) / 2.0
    shift_y = int(round(target_center - y_center))
    shift_x = int(round(target_center - x_center))
    return _shift_image(canvas, shift_y, shift_x)


# step 11 - scale pixels for the neural network
def normalize_for_model(img: np.ndarray) -> torch.Tensor:
    """Scale pixel values to [0, 1] and add the channel dimension the model expects."""

    return torch.from_numpy(img).to(dtype=torch.float32).div(255.0).unsqueeze(0)


# step 12 - run the full preprocessing pipeline and keep all stages
def build_preprocessing_stages(path: str | Path) -> PreprocessingStages:
    """Run the full image-cleaning pipeline and keep each intermediate result."""

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


# step 13 - final wrapper used by inference code
def preprocess_for_inference(path: str | Path) -> torch.Tensor:
    """Produce a batched tensor shaped exactly like the classifier expects."""

    stages = build_preprocessing_stages(path)
    tensor = normalize_for_model(stages.final_28x28)
    return tensor.unsqueeze(0)

#used in phase 7
def preprocess_crop_for_inference(crop:np.ndarray) ->torch.Tensor:
    """
    Preprocess a cropped symbol from segmentation for model inference.
    
    This mirrors the full preprocessing pipeline but assumes the input is already
    cropped from segmentation (thresholded binary image).
    
    Steps:
    1. Resize while preserving aspect ratio (longest side = 20 px)
    2. Center on 28x28 canvas
    3. Normalize to [0, 1]
    4. Add batch dimension
    
    Args:
        crop: Binary image from segmentation (white symbol on black background)
    
    Returns:
        Batched tensor (1, 1, 28, 28) ready for model inference
    """
    # Resize while preserving aspect ratio
    resized = resize_preserve_aspect(crop)
    # Center on 28x28 canvas
    centered = center_on_canvas(resized)
    # Normalize and add dimensions
    tensor = normalize_for_model(centered)
    return tensor.unsqueeze(0)
    