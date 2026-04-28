"""Operator recognition from cropped symbol images.

Takes cropped operator images from segmentation, preprocesses each one
to 28x28 MNIST format, and runs the operator CNN from Phase 8.
Output is a list of (operator_symbol, confidence) pairs in reading order.
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
from preprocessing.image_utils import preprocess_crop_for_inference
from recognition.operator_dataset import IDX_TO_CLASS

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "operator_cnn" / "best.pt"
OP_CONFIDENCE_THRESHOLD = 0.65


# step 1 - load the trained CNN from disk
def load_model(device: torch.device) -> MNISTCNN:
    """Rebuild the model from checkpoint config and load its saved weights."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = checkpoint.get("config", {})

    model = MNISTCNN(
        hidden_size=int(config.get("hidden_size", 64)),
        num_classes=int(config.get("num_classes", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# step 2 - classify a single cropped operator
def recognize_operator(crop, model: MNISTCNN) -> tuple[str, float]:
    h, w = crop.shape
    aspect = w / h if h > 0 else 0

    # minus sign is much wider than tall — catch it before the CNN
    if aspect > 2.2:
        return '-', 0.95

    inverted = cv2.bitwise_not(crop)
    _, cleaned = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
    tensor = preprocess_crop_for_inference(cleaned)
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)

    return IDX_TO_CLASS[int(predicted.item())], float(confidence.item())


# step 3 - classify every crop in reading order
def recognize_all(crops: list, model: MNISTCNN) -> list[tuple[str, float]]:
    """
    Run recognize_operator on every crop.
    Returns [(operator_symbol, confidence), ...] in left-to-right order.
    """
    return [recognize_operator(crop, model) for crop in crops]


if __name__ == "__main__":
    # single crop test
    crop = cv2.imread("artifacts/phase6/sample_0/crop_1.png", 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    op, conf = recognize_operator(crop, model)
    print(f"Prediction: {op}, Confidence: {conf:.4f}")