"""Run the trained robust CNN on the local handwritten sample images.

This is the end-to-end sanity check for phase 5:
take custom images, preprocess them into MNIST-like tensors, load the best
robust checkpoint, and write a small prediction report to the artifacts folder.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_mnist import MNISTCNN
from preprocessing.image_utils import preprocess_for_inference

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_digits"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cnn_robust_best.pt"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "phase5"
PREDICTIONS_CSV = ARTIFACTS_DIR / "predictions.csv"
PREDICTIONS_LOG = ARTIFACTS_DIR / "predictions.txt"
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


# step 1 - find the sample images we want to test
def sample_image_paths(sample_dir: Path) -> list[Path]:
    """Return the sample images we want to feed through the inference pipeline."""

    return sorted(
        path
        for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


# step 2 - rebuild the trained model from its checkpoint
def load_model(device: torch.device) -> MNISTCNN:
    """Rebuild the model from checkpoint config and load its saved weights."""

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = checkpoint.get("config", {})

    model = MNISTCNN(
        hidden_size=int(config.get("hidden_size", 128)),
        num_classes=int(config.get("num_classes", 10)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# step 3 - run end-to-end prediction on the sample folder
def main() -> None:
    """Predict labels for each sample image and save both text and csv outputs."""

    sample_paths = sample_image_paths(SAMPLE_DIR)
    if not sample_paths:
        raise FileNotFoundError(f"No sample images found under {SAMPLE_DIR}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # step 1 - load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    rows: list[dict[str, str | int | float]] = []
    log_lines: list[str] = []

    print(f"Found {len(sample_paths)} image(s) in {SAMPLE_DIR}")

    # step 2 - preprocess each image and ask the model for a prediction
    for image_path in sample_paths:
        tensor = preprocess_for_inference(image_path).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)

        pred_value = int(prediction.item())
        conf_value = float(confidence.item())
        line = f"{image_path.name} -> Pred: {pred_value} | Conf: {conf_value:.4f}"

        print(line)
        log_lines.append(line)
        rows.append(
            {
                "image": image_path.name,
                "predicted": pred_value,
                "confidence": round(conf_value, 4),
            }
        )

    # step 3 - save both machine-readable and human-readable outputs
    with PREDICTIONS_CSV.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image", "predicted", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)

    PREDICTIONS_LOG.write_text("\n".join(log_lines) + "\n")
    print(f"\nSaved prediction logs to {PREDICTIONS_LOG}")
    print(f"Saved prediction table to {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
