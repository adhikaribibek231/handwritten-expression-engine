"""Train the spatially-aware CNN on handwritten operators.

Trains a CNN to recognize +, -, ×, ÷ from cropped symbol images.
The model is smaller than the digit model since operator recognition
is a simpler 4-class problem compared to 10-digit MNIST.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_mnist import MNISTCNN
from recognition.operator_dataset import get_splits

# Config (Phase 8 Operator CNN)
SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64  # smaller than digit model — simpler 4-class problem
NUM_CLASSES = 4

DATA_DIR = PROJECT_ROOT / "data" / "operators"
METRICS_DIR = PROJECT_ROOT / "metrics"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "operator_cnn"
METRICS_FILE = METRICS_DIR / "operator_cnn.csv"
BEST_CKPT = CHECKPOINTS_DIR / "best.pt"
LAST_CKPT = CHECKPOINTS_DIR / "last.pt"


# step 1 - make the random pieces repeatable
def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# step 2 - save the training history after each epoch
def write_metrics_csv(rows: list[dict], path: Path) -> None:
    """Persist the per-epoch history so we can inspect learning later."""
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "seed",
        "batch_size",
        "learning_rate",
        "hidden_size",
        "num_classes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# step 3 - shared evaluation helper for val and test
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> tuple[float, float]:
    """Score a model on one loader without updating weights."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long()

            logits = model(x)
            loss = criterion(logits, y)

            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return loss_sum / total, correct / total


# step 4 - run the full operator CNN training pipeline
def main() -> None:
    """Train the operator CNN, save checkpoints, and report results."""

    # step 1 - prepare reproducibility and output folders
    set_seed(SEED)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # step 2 - load operator dataset and build train/val loaders
    train_set, val_set = get_splits(DATA_DIR)

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=use_cuda
    )

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    # step 3 - build the model and optimizer
    model = MNISTCNN(hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)
    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    history: list[dict] = []

    # step 4 - train, evaluate, and checkpoint after every epoch
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += x.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "hidden_size": HIDDEN_SIZE,
            "num_classes": NUM_CLASSES,
        }
        history.append(row)

        # Write logs every epoch so partial progress is never lost.
        write_metrics_csv(history, METRICS_FILE)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "seed": SEED,
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "hidden_size": HIDDEN_SIZE,
                        "num_classes": NUM_CLASSES,
                    },
                },
                BEST_CKPT,
            )

        print(
            f"Epoch {epoch:2d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

    # step 5 - save the last checkpoint
    torch.save(
        {
            "epoch": EPOCHS,
            "best_val_acc": best_val_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "hidden_size": HIDDEN_SIZE,
                "num_classes": NUM_CLASSES,
            },
        },
        LAST_CKPT,
    )
    best_ckpt  = torch.load(BEST_CKPT, map_location=device)
    best_epoch = int(best_ckpt["epoch"])
    best_model = MNISTCNN(hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)
    best_model.load_state_dict(best_ckpt["model_state_dict"])
    best_model.eval()
    best_val_loss, best_val_acc = evaluate(best_model, val_loader, criterion, device)

    print(f"\nBest epoch:     {best_epoch}")
    print(f"Best val acc:   {best_val_acc:.4f}")
    print(f"Metrics saved:  {METRICS_FILE}")
    print(f"Best checkpoint: {BEST_CKPT}")


if __name__ == "__main__":
    main()