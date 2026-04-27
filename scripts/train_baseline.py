"""Train the simple dense MNIST baseline.

this script is the plain starting point for the project:
download MNIST, split it into train and validation sets, train the MLP,
save metrics and checkpoints, then compare the last model with the best
validation checkpoint on the test set.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_dense import MNISTBaseline

# -----------------------------
# Config (Phase 2 baseline)
# -----------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
TRAIN_SIZE = 50_000
VAL_SIZE = 10_000
HIDDEN_SIZE = 128
NUM_CLASSES = 10

DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = PROJECT_ROOT / "metrics"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
METRICS_FILE = METRICS_DIR / "baseline_dense.csv"
BEST_CKPT = CHECKPOINTS_DIR / "baseline_dense_best.pt"
LAST_CKPT = CHECKPOINTS_DIR / "baseline_dense_last.pt"

# step 1 - make the random pieces repeatable
def set_seed(seed: int) -> None:
    """Lock down the random pieces so runs stay repeatable."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# step 2 - save the training history after each epoch
def write_metrics_csv(rows: list[dict], path: Path) -> None:
    """Write the accumulated training history to disk after each epoch."""

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
    """Run one full evaluation pass and return average loss plus accuracy."""

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

# step 4 - run the full baseline training pipeline
def main() -> None:
    """Run the full baseline experiment from data loading to saved artifacts."""

    # step 1 - prepare reproducibility and output folders
    set_seed(SEED)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # step 2 - load MNIST and build the train/val/test loaders
    # Load the full dataset once, then carve out the reproducible train/val split.
    # MNIST tensors come out as (1, 28, 28) and pixels in [0, 1].
    train_full = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=ToTensor())
    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=ToTensor())

    split_gen = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(train_full, [TRAIN_SIZE, VAL_SIZE], generator=split_gen)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # step 3 - build the model and optimizer
    model = MNISTBaseline(hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)
    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = float("-inf")
    history: list[dict] = []

    # step 4 - train epoch by epoch and keep enough logs to compare runs later
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).long()

            optimizer.zero_grad()
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
                    },
                },
                BEST_CKPT,
            )

        print(
            f"Epoch {epoch} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

    # step 5 - save the last checkpoint and compare it with the best checkpoint
    # Optional final checkpoint for exact end-of-run state.
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
            },
        },
        LAST_CKPT,
    )

    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)

    # Then check the checkpoint that actually won on validation, not just the last epoch.
    best_ckpt = torch.load(BEST_CKPT, map_location=device)
    best_epoch = int(best_ckpt["epoch"])
    best_model = MNISTBaseline(hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)
    best_model.load_state_dict(best_ckpt["model_state_dict"])
    best_test_loss, best_test_acc = evaluate(best_model, test_loader, criterion, device)

    print(f"Final val acc: {float(history[-1]['val_acc']):.4f}")
    print(f"Best val acc:  {best_val_acc:.4f}")
    print(f"Final test loss: {final_test_loss:.4f}")
    print(f"Final test acc:  {final_test_acc:.4f}")
    print(f"Best epoch:      {best_epoch}")
    print(f"Best-ckpt test loss: {best_test_loss:.4f}")
    print(f"Best-ckpt test acc:  {best_test_acc:.4f}")
    print(f"Metrics saved: {METRICS_FILE}")
    print(f"Best ckpt:     {BEST_CKPT}")


if __name__ == "__main__":
    main()
