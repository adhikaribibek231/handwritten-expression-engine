"""Tests for the segmentation pipeline.

These check the basic promises of vision/segmentation.py:
- images load correctly
- thresholding produces a valid binary image
- contour detection finds boxes with sane dimensions
- filtering removes small noise but keeps real symbols
- sorting puts boxes in left-to-right order
- the full pipeline returns the right shapes
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.segmentation import (
    load_expression_image,
    threshold_expression,
    find_symbol_boxes,
    filter_boxes,
    sort_boxes_left_to_right,
    crop_symbols,
    segment_expression,
)

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_expressions"


class TestSegmentation(unittest.TestCase):
    """Exercise each step of the segmentation pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Find a sample expression image to test against."""
        cls.sample_paths = sorted(SAMPLE_DIR.glob("sample_*.png"))
        if not cls.sample_paths:
            raise AssertionError(f"No sample expression images found under {SAMPLE_DIR}")
        cls.sample_path = cls.sample_paths[0]

    # step 1 - loading
    def test_load_returns_grayscale_array(self) -> None:
        """Loading an expression image should give a 2D numpy array."""
        img = load_expression_image(self.sample_path)
        self.assertEqual(len(img.shape), 2)
        self.assertEqual(img.dtype, np.uint8)

    def test_load_raises_on_missing_file(self) -> None:
        """Loading a nonexistent path should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_expression_image(Path("does_not_exist.png"))

    # step 2 - thresholding
    def test_threshold_produces_binary(self) -> None:
        """Thresholded image should only contain 0 and 255."""
        img = load_expression_image(self.sample_path)
        binary = threshold_expression(img)
        unique_values = set(np.unique(binary))
        self.assertTrue(unique_values.issubset({0, 255}))

    def test_threshold_same_shape(self) -> None:
        """Thresholded image should keep the same shape as the input."""
        img = load_expression_image(self.sample_path)
        binary = threshold_expression(img)
        self.assertEqual(img.shape, binary.shape)

    # step 3 - contour detection
    def test_find_boxes_returns_list_of_tuples(self) -> None:
        """Each box should be an (x, y, w, h) tuple with positive dimensions."""
        img = load_expression_image(self.sample_path)
        binary = threshold_expression(img)
        boxes = find_symbol_boxes(binary)
        self.assertGreater(len(boxes), 0)
        for box in boxes:
            self.assertEqual(len(box), 4)
            x, y, w, h = box
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)

    # step 4 - filtering
    def test_filter_removes_tiny_boxes(self) -> None:
        """Filtering should never add boxes, only remove them."""
        fake_boxes = [(0, 0, 2, 2), (10, 10, 50, 50), (20, 20, 3, 80)]
        filtered = filter_boxes(fake_boxes, min_w=5, min_h=5, min_area=150)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], (10, 10, 50, 50))

    def test_filter_keeps_big_boxes(self) -> None:
        """Large boxes should survive filtering."""
        big_boxes = [(0, 0, 30, 40), (50, 0, 25, 35)]
        filtered = filter_boxes(big_boxes)
        self.assertEqual(len(filtered), 2)

    # step 5 - sorting
    def test_sort_puts_leftmost_first(self) -> None:
        """After sorting, boxes should be in ascending x order."""
        boxes = [(100, 0, 20, 20), (5, 0, 20, 20), (50, 0, 20, 20)]
        sorted_boxes = sort_boxes_left_to_right(boxes)
        xs = [b[0] for b in sorted_boxes]
        self.assertEqual(xs, sorted(xs))

    # step 6 - cropping
    def test_crops_match_box_count(self) -> None:
        """Number of crops should match the number of boxes."""
        img = load_expression_image(self.sample_path)
        binary = threshold_expression(img)
        boxes = find_symbol_boxes(binary)
        boxes = filter_boxes(boxes)
        crops = crop_symbols(binary, boxes)
        self.assertEqual(len(crops), len(boxes))

    def test_crops_are_nonempty(self) -> None:
        """Each crop should have nonzero dimensions."""
        img = load_expression_image(self.sample_path)
        binary = threshold_expression(img)
        boxes = find_symbol_boxes(binary)
        boxes = filter_boxes(boxes)
        crops = crop_symbols(binary, boxes)
        for crop in crops:
            self.assertGreater(crop.shape[0], 0)
            self.assertGreater(crop.shape[1], 0)

    # step 7 - full pipeline
    def test_full_pipeline_returns_correct_types(self) -> None:
        """segment_expression should return (boxes, crops, binary)."""
        boxes, crops, binary = segment_expression(self.sample_path)
        self.assertIsInstance(boxes, list)
        self.assertIsInstance(crops, list)
        self.assertIsInstance(binary, np.ndarray)
        self.assertEqual(len(boxes), len(crops))

    def test_full_pipeline_finds_symbols(self) -> None:
        """A real expression image should contain at least one symbol."""
        boxes, crops, binary = segment_expression(self.sample_path)
        self.assertGreater(len(boxes), 0)

    def test_full_pipeline_on_all_samples(self) -> None:
        """Every sample expression should segment without errors."""
        for path in self.sample_paths:
            boxes, crops, binary = segment_expression(path)
            self.assertGreater(len(boxes), 0, f"No symbols found in {path.name}")


if __name__ == "__main__":
    unittest.main()
