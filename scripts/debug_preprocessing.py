"""Visualize every preprocessing stage for the handwritten sample digits.

The goal here is simple: make the pipeline easy to inspect with images.
For each sample we save the original input, every intermediate stage, and a
small overview grid of the final 28x28 outputs.
"""

from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.image_utils import PreprocessingStages, build_preprocessing_stages

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_digits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "phase5"
_RESAMPLING = getattr(Image, "Resampling", Image)

SAMPLE_DIR.mkdir(parents= True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents= True, exist_ok=True)

# step 1 - save one stage image to disk
def save_stage_image(img, path: Path) -> None:
    """Save one stage image, creating the parent directory if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype("uint8"), mode="L").save(path)


# step 2 - save every stage for one handwritten sample
def save_sample_stages(image_path: Path, stages: PreprocessingStages) -> None:
    """Write all saved preprocessing views for one input image."""

    sample_dir = ARTIFACTS_DIR / image_path.stem
    stage_images = {
        "original.png": stages.original,
        "contrast.png": stages.contrast,
        "thresholded.png": stages.thresholded,
        "cropped.png": stages.cropped,
        "resized.png": stages.resized,
        "centered.png": stages.centered,
        "final_28x28.png": stages.final_28x28,
    }

    for filename, img in stage_images.items():
        save_stage_image(img, sample_dir / filename)


# step 3 - build a simple contact sheet of the final outputs
def save_final_overview(sample_outputs: list[tuple[str, PreprocessingStages]]) -> None:
    """Build a simple contact sheet of the final 28x28 images."""

    columns = 5
    cell_size = 84
    padding = 8
    rows = (len(sample_outputs) + columns - 1) // columns

    height = rows * cell_size + (rows + 1) * padding
    width = columns * cell_size + (columns + 1) * padding
    canvas = Image.new("L", (width, height), color=0)

    for index, (_, stages) in enumerate(sample_outputs):
        row = index // columns
        col = index % columns
        x_offset = padding + col * (cell_size + padding)
        y_offset = padding + row * (cell_size + padding)

        tile = Image.fromarray(stages.final_28x28, mode="L").resize(
            (cell_size, cell_size),
            _RESAMPLING.NEAREST,
        )
        canvas.paste(tile, (x_offset, y_offset))

    canvas.save(ARTIFACTS_DIR / "final_28x28_overview.png")


# step 4 - run the preprocessing debugger end to end
def main() -> None:
    """Run the preprocessing debugger over the saved handwritten samples."""

    sample_paths = sorted(SAMPLE_DIR.glob("sample_*.jpeg"))
    if not sample_paths:
        raise FileNotFoundError(f"No sample images found under {SAMPLE_DIR}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    sample_outputs: list[tuple[str, PreprocessingStages]] = []

    # step 1 - save every intermediate image for each sample
    for image_path in sample_paths:
        stages = build_preprocessing_stages(image_path)
        save_sample_stages(image_path, stages)
        sample_outputs.append((image_path.name, stages))
        print(
            f"{image_path.name} -> saved {ARTIFACTS_DIR / image_path.stem} "
            f"(threshold={stages.threshold}, bbox={stages.bbox})"
        )

    # step 2 - save the final overview grid
    save_final_overview(sample_outputs)
    print(f"Saved overview image to {ARTIFACTS_DIR / 'final_28x28_overview.png'}")


if __name__ == "__main__":
    main()
