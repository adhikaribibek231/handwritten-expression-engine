from __future__ import annotations

import sys
from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_mnist import MNISTCNN

DATA_DIR = PROJECT_ROOT / "data"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "cnn_robust_best.pt"
OUT_DIR = PROJECT_ROOT / "artifacts" / "phase4"
CSV_PATH = PROJECT_ROOT / "metrics" / "robust_failure_analysis.csv"

BATCH_SIZE = 256
NUM_CLASSES = 10
HIDDEN_SIZE = 128
TOP_K_GALLERY = 36
TOP_K_CONFUSIONS = 10


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Robust CNN Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(NUM_CLASSES))
    plt.yticks(range(NUM_CLASSES))
    plt.colorbar()

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def save_gallery(samples: list[dict], out_path: Path, title_prefix: str) -> None:
    if len(samples) == 0:
        print(f"No samples for gallery: {out_path.name}")
        return

    n = min(len(samples), TOP_K_GALLERY)
    grid = int(np.ceil(np.sqrt(n)))

    plt.figure(figsize=(10, 10))
    for idx in range(n):
        sample = samples[idx]
        img = sample["image"].squeeze(0).numpy()

        plt.subplot(grid, grid, idx + 1)
        plt.imshow(img, interpolation="nearest")
        plt.axis("off")
        plt.title(
            f'T:{sample["true_label"]} P:{sample["pred_label"]}\nC:{sample["confidence"]:.3f}',
            fontsize=8,
        )

    plt.suptitle(title_prefix, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def save_confidence_hist(correct_samples: list[dict], mis_samples: list[dict], out_path: Path) -> None:
    correct_conf = [s["confidence"] for s in correct_samples]
    wrong_conf = [s["confidence"] for s in mis_samples]

    plt.figure(figsize=(8, 5))
    plt.hist(correct_conf, bins=30, alpha=0.7, label="Correct")
    plt.hist(wrong_conf, bins=30, alpha=0.7, label="Wrong")
    plt.xlabel("Max softmax confidence")
    plt.ylabel("Count")
    plt.title("Confidence Histogram: Correct vs Wrong")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def save_top_confusions(cm: np.ndarray, out_path: Path, top_k: int = TOP_K_CONFUSIONS) -> None:
    pairs = []
    for true_label in range(NUM_CLASSES):
        for pred_label in range(NUM_CLASSES):
            if true_label == pred_label:
                continue
            count = int(cm[true_label, pred_label])
            if count > 0:
                pairs.append((count, true_label, pred_label))

    pairs.sort(reverse=True)

    with out_path.open("w") as f:
        f.write("Top confusion pairs (true -> predicted)\n")
        f.write("----------------------------------------\n")
        for count, true_label, pred_label in pairs[:top_k]:
            f.write(f"True {true_label} -> Pred {pred_label}: {count}\n")

    print(f"Saved: {out_path}")


def save_per_class_accuracy(cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    per_class_total = cm.sum(axis=1)
    per_class_correct = np.diag(cm)
    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)

    with out_path.open("w") as f:
        f.write("Per-class accuracy (test set)\n")
        f.write("----------------------------------\n")
        for c in range(NUM_CLASSES):
            f.write(
                f"class {c}: {per_class_acc[c]:.4f} "
                f"(correct {per_class_correct[c]}/ total {per_class_total[c]})\n"
            )
        f.write("----------------------------------\n")
        f.write(f"overall: {(y_true == y_pred).mean():.4f}\n")

    print(f"Saved: {out_path}")


def save_failure_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    correct_samples: list[dict],
    mis_samples: list[dict],
    out_path: Path,
) -> None:
    overall_acc = float((y_true == y_pred).mean())
    mean_correct_conf = float(np.mean([s["confidence"] for s in correct_samples])) if correct_samples else 0.0
    mean_wrong_conf = float(np.mean([s["confidence"] for s in mis_samples])) if mis_samples else 0.0

    with out_path.open("w") as f:
        f.write("Robust Failure Analysis Summary\n")
        f.write("----------------------------------\n")
        f.write(f"Total samples: {len(y_true)}\n")
        f.write(f"Correct samples: {len(correct_samples)}\n")
        f.write(f"Wrong samples: {len(mis_samples)}\n")
        f.write(f"Accuracy: {overall_acc:.4f}\n")
        f.write(f"Mean confidence (correct): {mean_correct_conf:.4f}\n")
        f.write(f"Mean confidence (wrong): {mean_wrong_conf:.4f}\n")

    print(f"Saved: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = MNISTCNN(hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_true: list[int] = []
    all_pred: list[int] = []

    mis_samples: list[dict] = []
    correct_samples: list[dict] = []

    global_index = 0
    csv_rows =[]
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device).long()

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            max_conf, preds = probs.max(dim=1)

            x_cpu = x.cpu()
            y_cpu = y.cpu()
            preds_cpu = preds.cpu()
            max_conf_cpu = max_conf.cpu()
            probs_cpu = probs.cpu()

            all_true.extend(y_cpu.tolist())
            all_pred.extend(preds_cpu.tolist())

            batch_size = x.size(0)
            
            for i in range(batch_size):
                sample = {
                    "index": global_index,
                    "image": x_cpu[i],
                    "true_label": int(y_cpu[i].item()),
                    "pred_label": int(preds_cpu[i].item()),
                    "confidence": float(max_conf_cpu[i].item()),
                }
            
                if sample["true_label"] == sample["pred_label"]:
                    correct_samples.append(sample)
                else:
                    mis_samples.append(sample)

                global_index += 1

                row = {
                    "index": sample["index"],
                    "true_label": sample["true_label"],
                    "pred_label": sample["pred_label"],
                    "is_correct": int(sample["true_label"] == sample["pred_label"]),
                    "confidence": sample["confidence"],
                }
                for j in range(NUM_CLASSES):
                    row[f"prob_{j}"] = float(probs_cpu[i, j].item())
                csv_rows.append(row)


    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["index", "true_label", "pred_label", "is_correct", "confidence"] + [
        f"prob_{i}" for i in range(NUM_CLASSES)
    ]

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved: {CSV_PATH}")

    y_true = np.array(all_true, dtype=np.int64)
    y_pred = np.array(all_pred, dtype=np.int64)

    cm = confusion_matrix_np(y_true, y_pred, NUM_CLASSES)

    save_confusion_matrix(cm, OUT_DIR / "robust_confusion_matrix.png")
    save_per_class_accuracy(cm, y_true, y_pred, OUT_DIR / "robust_per_class_accuracy.txt")
    save_top_confusions(cm, OUT_DIR / "robust_top_confusions.txt", top_k=TOP_K_CONFUSIONS)
    save_failure_summary(
        y_true,
        y_pred,
        correct_samples,
        mis_samples,
        OUT_DIR / "robust_failure_summary.txt",
    )

    mis_samples.sort(key=lambda s: s["confidence"], reverse=True)
    correct_samples.sort(key=lambda s: s["confidence"])

    save_gallery(
        mis_samples[:TOP_K_GALLERY],
        OUT_DIR / "robust_high_confidence_errors.png",
        "High-Confidence Wrong Predictions",
    )
    save_gallery(
        correct_samples[:TOP_K_GALLERY],
        OUT_DIR / "robust_low_confidence_correct.png",
        "Low-Confidence Correct Predictions",
    )
    save_confidence_hist(
        correct_samples,
        mis_samples,
        OUT_DIR / "robust_confidence_hist.png",
    )


if __name__ == "__main__":
    main()
