"""Tests for the handwritten-digit preprocessing pipeline.

These checks are not trying to prove the images look perfect.
they just make sure the pipeline keeps the important promises:
the output is model-shaped, normalized, non-empty, and reasonably centered
for both the local handwritten samples and rendered MNIST digits.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.image_utils import build_preprocessing_stages, preprocess_for_inference

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample_digits"
_RESAMPLING = getattr(Image, "Resampling", Image)


# step 1 - helper for checking whether final digits are centered
def center_of_mass(img: np.ndarray) -> tuple[float, float]:
    """Compute the weighted center of an image for centering assertions."""

    weights = img.astype(np.float64)
    total = weights.sum()
    ys, xs = np.indices(img.shape)
    return float((ys * weights).sum() / total), float((xs * weights).sum() / total)


# step 2 - make MNIST digits look more like custom handwritten inputs
def render_mnist_for_inference(image: Image.Image) -> Image.Image:
    """Turn an MNIST digit into a larger white-page sample for preprocessing tests."""

    digit = np.array(image, dtype=np.uint8)
    inverted = 255 - digit

    rendered_digit = Image.fromarray(inverted, mode="L").resize(
        (56, 56),
        _RESAMPLING.NEAREST,
    )
    canvas = Image.new("L", (112, 112), color=255)
    offset = (
        (canvas.width - rendered_digit.width) // 2,
        (canvas.height - rendered_digit.height) // 2,
    )
    canvas.paste(rendered_digit, offset)
    return canvas


class TestImageUtils(unittest.TestCase):
    """Exercise the preprocessing pipeline on both real samples and MNIST-derived ones."""

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare handwritten samples and one rendered MNIST sample per digit."""

        # step 1 - collect the local handwritten samples
        cls.handwritten_paths = sorted(SAMPLE_DIR.glob("sample_*.jpeg"))
        if not cls.handwritten_paths:
            raise AssertionError(f"No sample images found under {SAMPLE_DIR}")

        # step 2 - build temporary rendered MNIST samples so all digits are covered
        cls.temp_dir = tempfile.TemporaryDirectory()
        temp_root = Path(cls.temp_dir.name)

        dataset = datasets.MNIST(root=DATA_DIR, train=False, download=False)
        cls.mnist_paths: list[Path] = []
        seen_digits: set[int] = set()

        for image, label in dataset:
            if label in seen_digits:
                continue

            rendered = render_mnist_for_inference(image)
            output_path = temp_root / f"mnist_{label}.png"
            rendered.save(output_path)
            cls.mnist_paths.append(output_path)
            seen_digits.add(label)

            if len(seen_digits) == 10:
                break

        if len(cls.mnist_paths) != 10:
            raise AssertionError("Could not build one rendered MNIST sample per digit.")

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up the temporary rendered images created for the tests."""

        cls.temp_dir.cleanup()

    # step 3 - shared assertion for final model tensor shape and range
    def assert_tensor_properties(self, path: Path) -> None:
        """Check the tensor contract expected by the classifier."""

        tensor = preprocess_for_inference(path)

        self.assertEqual(tensor.shape, (1, 1, 28, 28), path.name)
        self.assertGreaterEqual(float(tensor.min().item()), 0.0, path.name)
        self.assertLessEqual(float(tensor.max().item()), 1.0, path.name)
        self.assertGreater(float(tensor.sum().item()), 0.0, path.name)

    # step 4 - shared assertion for stage-level geometry checks
    def assert_stage_properties(self, path: Path) -> None:
        """Check that the final image is non-empty and centered well enough."""

        stages = build_preprocessing_stages(path)

        self.assertEqual(stages.final_28x28.shape, (28, 28), path.name)
        self.assertGreater(int(stages.thresholded.sum()), 0, path.name)

        y_center, x_center = center_of_mass(stages.final_28x28)
        self.assertLess(abs(y_center - 13.5), 2.5, path.name)
        self.assertLess(abs(x_center - 13.5), 2.5, path.name)

    def test_preprocess_tensor_shape_and_range_for_handwritten_samples(self) -> None:
        """Handwritten samples should become valid normalized model inputs."""

        for path in self.handwritten_paths:
            self.assert_tensor_properties(path)

    def test_final_stage_is_28x28_and_centered_for_handwritten_samples(self) -> None:
        """Handwritten samples should end up centered on a 28x28 canvas."""

        for path in self.handwritten_paths:
            self.assert_stage_properties(path)

    def test_preprocess_tensor_shape_and_range_for_rendered_mnist_samples(self) -> None:
        """Rendered MNIST samples should also satisfy the model-input contract."""

        for path in self.mnist_paths:
            self.assert_tensor_properties(path)

    def test_final_stage_is_28x28_and_centered_for_rendered_mnist_samples(self) -> None:
        """Rendered MNIST samples should remain centered after preprocessing."""

        for path in self.mnist_paths:
            self.assert_stage_properties(path)


if __name__ == "__main__":
    unittest.main()
