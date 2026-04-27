"""
Answers the question: "Did the segmentation work correctly?"
saves to artifacts/phase6/<expression_name>/:
    original.png - the raw input image
    thresholded.png - binary image after thresholding
    boxed_overlay.png - original with bounding boxes drawn
    crop_0.png - first symbol crop
    crop_1.png - second symbol crop

usage:
    python scripts/debug_segmentation.py
    python scripts/debug_segmentation.py data/sample_expressions/12+7.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.segmentation import segment_expression

# step 1 - config
SAMPLE_IMAGE = PROJECT_ROOT / "data" / "sample_expressions" / "sample_0.png"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "phase6"
 
 
# step 2 - save helpers

def save_original(image_path: Path, out_dir: Path) -> None:
    """Save a copy of the raw input image."""
    img = cv2.imread(str(image_path))
    cv2.imwrite(str(out_dir / 'original.png'), img)
    print(f"  [saved] original.png")
 
 
def save_thresholded(binary, out_dir: Path) -> None:
    """Save the binary image produced by thresholding."""
    cv2.imwrite(str(out_dir / 'thresholded.png'), binary)
    print(f"  [saved] thresholded.png")
 
 
def save_boxed_overlay(image_path: Path, boxes, out_dir: Path) -> None:
    """
    Draw bounding boxes on the original image and save.
 
    Each box is drawn in green with its index number above it.
    This is the most important debug image — inspect it carefully.
    """
    overlay = cv2.imread(str(image_path))
 
    for i, (x, y, w, h) in enumerate(boxes):
        # green rectangle around each detected symbol
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 80), 2)
 
        # index label above the box
        cv2.putText(
            overlay,
            str(i),
            (x, max(y - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 80),
            1,
            cv2.LINE_AA,
        )
 
    cv2.imwrite(str(out_dir / 'boxed_overlay.png'), overlay)
    print(f"  [saved] boxed_overlay.png  ({len(boxes)} boxes)")
 
 
def save_crops(crops, out_dir: Path) -> None:
    """Save each individual symbol crop as crop_N.png."""
    for i, crop in enumerate(crops):
        filename = f'crop_{i}.png'
        cv2.imwrite(str(out_dir / filename), crop)
        print(f"  [saved] {filename}  (shape: {crop.shape})")
 
 
# step 3 - run the segmentation debugger

def main() -> None:
    """Debug the segmentation pipeline on a handwritten expression image."""

    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_IMAGE

    print(f"\nDebugging: {image_path}")
    print("-" * 40)

    if not image_path.exists():
        print(f"ERROR: File not found — {image_path}")
        print("Put a handwritten expression image there first.")
        sys.exit(1)

    # run the full segmentation pipeline
    boxes, crops, binary = segment_expression(image_path)

    print(f"  Found {len(boxes)} symbol(s)")
    for i, (x, y, w, h) in enumerate(boxes):
        print(f"    [{i}]  x={x}  y={y}  w={w}  h={h}  area={w*h}")

    # save all debug artifacts
    out_dir = ARTIFACTS_DIR / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {out_dir}/")

    save_original(image_path, out_dir)
    save_thresholded(binary, out_dir)
    save_boxed_overlay(image_path, boxes, out_dir)
    save_crops(crops, out_dir)

    print(f"\nDone. Open boxed_overlay.png to verify correctness.")
    print(f"Each green box should contain exactly one symbol.")
 
 
if __name__ == "__main__":
    main()