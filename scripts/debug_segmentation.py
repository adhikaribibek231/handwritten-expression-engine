"""
Answers the question: "Did the segmentation work correctly?"
saves to artifacts/phase6/<expression_name>/:
    original.png - the raw input image
    thresholded.png - binary image after thresholding
    boxed_overlay.png - original with bounding boxes drawn
    crop_0.png - first symbol crop
    crop_1.png - second symbol crop

usage: 
    pyhon scripts/debug_segmentation.py
    python scripts/debug_segmentation.py data/sample_expressions/12+7.png
"""
import sys
import os
import cv2
 
# Make sure project root is on the path regardless of where script is run from
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
 
from pathlib import Path
from vision.segmentation import segment_expression
 
 
# ── Config ────────────────────────────────────────────────────────────────────
 
DEFAULT_IMAGE = Path('data/sample_expressions/sample_expr_1.png')
ARTIFACTS_DIR = Path('artifacts/phase6')
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def make_output_dir(image_path: Path) -> Path:
    """Create and return the output directory for this expression."""
    name = image_path.stem          # e.g. "12+7" from "12+7.png"
    out  = ARTIFACTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    return out
 
 
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
        # Green rectangle around each detected symbol
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 80), 2)
 
        # Index label above the box
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
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def debug(image_path: Path) -> None:
    print(f"\nDebugging: {image_path}")
    print("-" * 40)
 
    if not image_path.exists():
        print(f"ERROR: File not found — {image_path}")
        print("Put a handwritten expression image there first.")
        sys.exit(1)
 
    # Run the full segmentation pipeline
    boxes, crops, binary = segment_expression(image_path)
 
    print(f"  Found {len(boxes)} symbol(s)")
    for i, (x, y, w, h) in enumerate(boxes):
        print(f"    [{i}]  x={x}  y={y}  w={w}  h={h}  area={w*h}")
 
    # Save all debug artifacts
    out_dir = make_output_dir(image_path)
    print(f"\nSaving to: {out_dir}/")
 
    save_original(image_path, out_dir)
    save_thresholded(binary, out_dir)
    save_boxed_overlay(image_path, boxes, out_dir)
    save_crops(crops, out_dir)
 
    print(f"\nDone. Open boxed_overlay.png to verify correctness.")
    print(f"Each green box should contain exactly one symbol.")
 
 
if __name__ == '__main__':
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    debug(image_path)
 