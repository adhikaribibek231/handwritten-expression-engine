"""Analyze the plain CNN checkpoint on the MNIST test set.

The script does three things:
load the best saved model, run it on the test split, then turn the predictions
into a confusion matrix, a per-class report, and a gallery of mistakes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_mnist import MNISTCNN

# Paths
DATA_DIR = PROJECT_ROOT / "data"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "cnn_mnist" / "best.pt"
OUT_DIR = PROJECT_ROOT / "artifacts" / "phase3"

BATCH_SIZE = 256
NUM_CLASSES = 10
HIDDEN_SIZE = 128


# step 1 - build a confusion matrix from raw labels
def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Build a confusion matrix using NumPy only."""

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# step 2 - run the saved model and create phase-3 analysis artifacts
def main() -> None:
    """Run the saved CNN over the test set and write the phase-3 analysis artifacts."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # step 1 - load the test data and the trained checkpoint
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=use_cuda)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = MNISTCNN(hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # step 2 - collect predictions and keep a small sample of mistakes
    all_true: list[int] = []
    all_pred: list[int] = []
    mis_imgs: list[torch.Tensor] = []
    mis_true: list[int] = []
    mis_pred: list[int] = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device).long()

            logits = model(x)
            preds = logits.argmax(dim=1)

            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

            mism = preds != y
            if mism.any() and len(mis_imgs) < 36:
                xm = x[mism].cpu()
                yt = y[mism].cpu().tolist()
                yp = preds[mism].cpu().tolist()

                for i in range(xm.size(0)):
                    if len(mis_imgs) >= 36:
                        break
                    mis_imgs.append(xm[i])
                    mis_true.append(yt[i])
                    mis_pred.append(yp[i])

    y_true = np.array(all_true, dtype=np.int64)
    y_pred = np.array(all_pred, dtype=np.int64)

    # step 3 - save the confusion matrix and per-class numbers
    cm = confusion_matrix_np(y_true, y_pred, NUM_CLASSES)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("MNIST CNN Confusion Matrix(test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(NUM_CLASSES))
    plt.yticks(range(NUM_CLASSES))
    plt.colorbar()

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    cm_path = OUT_DIR / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved: {cm_path}")

    per_class_total = cm.sum(axis=1)
    per_class_correct = np.diag(cm)
    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)

    pca_path = OUT_DIR / "per_class_accuracy.txt"
    with pca_path.open("w") as f:
        f.write("Per-class accuracy (test set)\n")
        f.write("----------------------------------")
        for c in range(NUM_CLASSES):
            f.write(
                f"class {c}: {per_class_acc[c]: .4f} "
                f"(correct {per_class_correct[c]}/ total {per_class_total[c]})\n"
            )
        f.write("----------------------------------")
        f.write(f"overall: {(y_true == y_pred).mean():.4f}\n")

    print(f"Saved: {pca_path}")

    # step 4 - save a quick gallery of misclassified digits
    if len(mis_imgs) == 0:
        print("No misclassifications found (unexpected for MNIST). Skipping gallery.")
        return

    n = len(mis_imgs)
    grid = int(np.ceil(np.sqrt(n)))
    plt.figure(figsize=(10, 10))

    for idx in range(n):
        img = mis_imgs[idx].squeeze(0)  # (28, 28)
        plt.subplot(grid, grid, idx + 1)
        plt.imshow(img, interpolation="nearest")
        plt.axis("off")
        plt.title(f"T:{mis_true[idx]} P:{mis_pred[idx]}", fontsize=9)

    gallery_path = OUT_DIR / "misclassified.png"
    plt.tight_layout()
    plt.savefig(gallery_path, dpi=200)
    plt.close()
    print(f"Saved: {gallery_path}")


if __name__ == "__main__":
    main()
