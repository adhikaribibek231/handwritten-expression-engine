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

from models.cnn_mnist import MNISTCNN 

# -----------------------------
# Config (Phase 4 CNN)
# -----------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
TRAIN_SIZE = 50_000
VAL_SIZE = 10_000
HIDDEN_SIZE = 128
NUM_CLASSES = 10

DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = PROJECT_ROOT / "metrics"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
METRICS_FILE = METRICS_DIR / "cnn_robust.csv"
BEST_CKPT = CHECKPOINTS_DIR / "cnn_robust_best.pt"
LAST_CKPT = CHECKPOINTS_DIR / "cnn_robust_last.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_metrics_csv(rows: list[dict], path: Path) -> None:
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
        "train_size",
        "val_size",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model: torch.nn.Module, loader: DataLoader, criterion, device: torch.device) -> tuple[float, float]:
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


def main() -> None:
    set_seed(SEED)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # MNIST tensors come out as (1, 28, 28) and pixels in [0, 1].
    train_full = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=ToTensor())
    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=ToTensor())

    split_gen = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(train_full, [TRAIN_SIZE, VAL_SIZE], generator=split_gen)

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=use_cuda)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=use_cuda)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=use_cuda)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    model = MNISTCNN(hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)
    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    history: list[dict] = []

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
            "train_size": TRAIN_SIZE,
            "val_size": VAL_SIZE,
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
                        "train_size": TRAIN_SIZE,
                        "val_size": VAL_SIZE,
                    },
                },
                BEST_CKPT,
            )

        print(
            f"Epoch {epoch} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

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
                "num_classes": NUM_CLASSES,
                "train_size": TRAIN_SIZE,
                "val_size": VAL_SIZE,
            },
        },
        LAST_CKPT,
    )

    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)

    # Evaluate the checkpoint that achieved best validation accuracy.
    best_ckpt = torch.load(BEST_CKPT, map_location=device)
    best_epoch = int(best_ckpt["epoch"])
    best_model = MNISTCNN(hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)
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
