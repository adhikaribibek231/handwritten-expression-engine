"""Debug script for operator dataset validation.

Inspects the operator dataset for:
- Class distribution balance across +, -, ×, ÷
- Visual samples of operators to verify correct loading

Output: artifacts/phase8/sample_*.png files
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognition.operator_dataset import IDX_TO_CLASS, OperatorDataset

# Config
DATA_ROOT = PROJECT_ROOT / "data" / "operators"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "phase8"

# Symbol to filename mapping
SYMBOL_TO_FILENAME = {
    '+': 'add',
    '-': 'sub',
    '×': 'mul',
    '÷': 'div',
}


def get_symbol_filename(symbol: str) -> str:
    """Convert operator symbol to ASCII-safe filename."""
    return SYMBOL_TO_FILENAME.get(symbol, symbol)


def main() -> None:
    """Inspect operator dataset and save sample images."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = OperatorDataset(DATA_ROOT)

    # 1. Print class distribution
    counts = Counter(label for _, label in dataset.samples)
    print("\nClass distribution:")
    for idx, count in sorted(counts.items()):
        symbol = IDX_TO_CLASS[idx]
        print(f"  {symbol} (class {idx}): {count} samples")

    # 2. Save one sample per class for visual inspection
    print("\nSaving sample images to artifacts/phase8/...")
    seen = set()
    for img_tensor, label in dataset:
        if label in seen:
            continue
        seen.add(label)
        # tensor → numpy for saving
        arr = (img_tensor.squeeze().numpy() * 0.5 + 0.5) * 255
        arr = arr.astype(np.uint8)
        symbol = IDX_TO_CLASS[label]
        filename = f"sample_{get_symbol_filename(symbol)}.png"
        cv2.imwrite(str(ARTIFACTS_DIR / filename), arr)
        if len(seen) == 4:
            break

    print("Done. Open artifacts/phase8/ and verify each sample looks correct.")


if __name__ == "__main__":
    main()