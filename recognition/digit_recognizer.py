from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from models.cnn_mnist import MNISTCNN
from vision.segmentation import segment_expression
from preprocessing.image_utils import preprocess_crop_for_inference
import cv2


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cnn_robust_best.pt"
CONFIDENCE_THRESHOLD = 0.85


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

def recognize_digit(crop, model) -> tuple[int, float]:
    """
    takes one cropped symbol image (numpy array).
    returns (predicted_digit, confidence)
    """
    tensor = preprocess_crop_for_inference(crop)
    device = next(model.parameters()).device  # Get device from model parameters
    tensor = tensor.to(device)  # Move tensor to same device as model
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)
    return int(predicted.item()), float(confidence.item())

def recognize_all(crops,model) -> list[tuple[int, float]]:
    """
    run recognize_digit on every crop.
    returns [(digit,confidence), ...] in left-to-right order
    """
    return [recognize_digit(crop, model) for crop in crops]

if __name__ == '__main__':
    crop = cv2.imread('artifacts/phase6/sample_0/crop_0.png', 0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)

    # single crop test
    digit, conf = recognize_digit(crop, model)
    print(f"Prediction: {digit}, Confidence: {conf:.4f}")

    # full expression test
    boxes, crops, binary = segment_expression('data/sample_expressions/sample_0.png')
    results = recognize_all(crops, model)
    print(results)

