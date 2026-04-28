"""Digit recognition from cropped symbol images.

Takes cropped symbol images from segmentation, preprocesses each one
to look like MNIST, and runs the robust CNN from Phase 4.
Output is a list of (digit, confidence) pairs in reading order.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import cv2

from models.cnn_mnist import MNISTCNN
from vision.segmentation import segment_expression
from preprocessing.image_utils import preprocess_crop_for_inference

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cnn_robust" / "best.pt"



# step 1 - load the trained CNN from disk
def load_model(device: torch.device) -> MNISTCNN:
    """Rebuild the model from checkpoint config and load its saved weights."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = checkpoint.get("config", {})

    model = MNISTCNN(
        hidden_size=int(config.get("hidden_size", 128)),
        num_classes=int(config.get("num_classes", 10)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# step 2 - classify a single cropped symbol
def recognize_digit(crop, model: MNISTCNN) -> tuple[int, float]:
    """
    Takes one cropped symbol image (numpy array).
    Returns (predicted_digit, confidence).
    """

    # preprocess: resize, center on 28x28 canvas, normalise to [0,1]
    tensor = preprocess_crop_for_inference(crop)

    # move to the same device the model lives on
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    # forward pass with no gradient tracking
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)

    return int(predicted.item()), float(confidence.item())


# step 3 - classify every crop in reading order
def recognize_all(crops: list, model: MNISTCNN) -> list[tuple[int, float]]:
    """
    Run recognize_digit on every crop.
    Returns [(digit, confidence), ...] in left-to-right order.
    """
    return [recognize_digit(crop, model) for crop in crops]


if __name__ == "__main__":
    # single crop test
    crop = cv2.imread("artifacts/phase6/sample_0/crop_0.png", 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    digit, conf = recognize_digit(crop, model)
    print(f"Prediction: {digit}, Confidence: {conf:.4f}")

    # full expression test
    boxes, crops, binary = segment_expression(
        "data/sample_expressions/sample_0.png",
    )
    results = recognize_all(crops, model)
    print(results)
