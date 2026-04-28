"""Tests for the recognition and grouping pipeline.

These check the promises of recognition/digit_recognizer.py and
recognition/grouping.py:
- the model loads from checkpoint without errors
- single-crop recognition returns (digit, confidence) with valid ranges
- batch recognition preserves crop count
- confidence checking correctly flags low values
- digit grouping merges sequences into numbers
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from recognition.digit_recognizer import load_model, recognize_digit, recognize_all
from recognition.grouping import group_digits, is_low_confidence, CONFIDENCE_THRESHOLD
from vision.segmentation import segment_expression

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_expressions"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "cnn_robust" / "best.pt"


class TestDigitRecognizer(unittest.TestCase):
    """Exercise the digit recognition pipeline on real crops."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the model and segment one sample expression for testing."""
        if not CHECKPOINT.exists():
            raise unittest.SkipTest(f"Checkpoint not found: {CHECKPOINT}")

        sample_paths = sorted(SAMPLE_DIR.glob("sample_*.png"))
        if not sample_paths:
            raise unittest.SkipTest(f"No sample expressions under {SAMPLE_DIR}")

        cls.device = torch.device("cpu")
        cls.model = load_model(cls.device)
        cls.sample_path = sample_paths[0]

        boxes, crops, binary = segment_expression(cls.sample_path)
        cls.crops = crops
        cls.boxes = boxes

    # step 1 - model loading
    def test_model_loads_in_eval_mode(self) -> None:
        """The loaded model should be in eval mode."""
        self.assertFalse(self.model.training)

    # step 2 - single crop recognition
    def test_recognize_single_crop(self) -> None:
        """Recognizing one crop should return (int, float)."""
        if not self.crops:
            self.skipTest("No crops from sample segmentation")

        digit, confidence = recognize_digit(self.crops[0], self.model)
        self.assertIsInstance(digit, int)
        self.assertIsInstance(confidence, float)

    def test_digit_in_valid_range(self) -> None:
        """Predicted digit should be 0-9."""
        if not self.crops:
            self.skipTest("No crops from sample segmentation")

        digit, _ = recognize_digit(self.crops[0], self.model)
        self.assertGreaterEqual(digit, 0)
        self.assertLessEqual(digit, 9)

    def test_confidence_in_valid_range(self) -> None:
        """Confidence should be between 0 and 1."""
        if not self.crops:
            self.skipTest("No crops from sample segmentation")

        _, confidence = recognize_digit(self.crops[0], self.model)
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    # step 3 - batch recognition
    def test_recognize_all_matches_crop_count(self) -> None:
        """recognize_all should return one result per crop."""
        results = recognize_all(self.crops, self.model)
        self.assertEqual(len(results), len(self.crops))

    def test_recognize_all_empty_input(self) -> None:
        """Empty crop list should give empty results."""
        results = recognize_all([], self.model)
        self.assertEqual(results, [])

    # step 4 - works on a synthetic crop (blank image)
    def test_recognize_blank_crop(self) -> None:
        """A blank (all-black) crop should still return a valid result."""
        blank = np.zeros((28, 28), dtype=np.uint8)
        digit, confidence = recognize_digit(blank, self.model)
        self.assertIsInstance(digit, int)
        self.assertGreaterEqual(digit, 0)
        self.assertLessEqual(digit, 9)


class TestGrouping(unittest.TestCase):
    """Exercise the grouping and confidence validation logic."""

    # step 1 - confidence checking
    def test_low_confidence_detects_bad_digit(self) -> None:
        """Should return True if any digit is below threshold."""
        results = [(1, 0.99), (2, 0.50), (3, 0.98)]
        self.assertTrue(is_low_confidence(results))

    def test_low_confidence_passes_good_digits(self) -> None:
        """Should return False when all digits are above threshold."""
        results = [(1, 0.99), (2, 0.95), (3, 0.98)]
        self.assertFalse(is_low_confidence(results))

    def test_low_confidence_empty_list(self) -> None:
        """Empty results should not flag as low confidence."""
        self.assertFalse(is_low_confidence([]))

    def test_low_confidence_custom_threshold(self) -> None:
        """Custom threshold should be respected."""
        results = [(1, 0.90)]
        self.assertFalse(is_low_confidence(results, threshold=0.85))
        self.assertTrue(is_low_confidence(results, threshold=0.95))

    # step 2 - digit grouping
    def test_group_single_digit(self) -> None:
        """One digit should give a one-element list."""
        self.assertEqual(group_digits([(7, 0.99)]), [7])

    def test_group_multi_digit(self) -> None:
        """Multiple digits should merge into one number."""
        self.assertEqual(group_digits([(1, 0.99), (2, 0.97), (3, 0.95)]), [123])

    def test_group_with_leading_zero(self) -> None:
        """Leading zeros collapse (0, 5 -> 5, not 05)."""
        self.assertEqual(group_digits([(0, 0.99), (5, 0.98)]), [5])

    def test_group_empty(self) -> None:
        """Empty input should give empty output."""
        self.assertEqual(group_digits([]), [])

    def test_group_single_zero(self) -> None:
        """Just a zero should give [0]."""
        self.assertEqual(group_digits([(0, 0.99)]), [0])

    def test_group_large_number(self) -> None:
        """Should handle large multi-digit numbers."""
        digits = [(d, 0.99) for d in [9, 8, 7, 6, 5]]
        self.assertEqual(group_digits(digits), [98765])


if __name__ == "__main__":
    unittest.main()
