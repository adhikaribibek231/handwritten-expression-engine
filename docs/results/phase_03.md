# Phase 03 Results — CNN Perception Model

## Objective

Train and evaluate a CNN-based MNIST digit recognizer and document error patterns.

## Experiment Setup

- Training script: `scripts/train_cnn.py`
- Analysis script: `scripts/analyze_cnn.py`
- Model: `models/cnn_mnist.py`
- Dataset: MNIST
- Split: train `50,000` / val `10,000` / test `10,000`
- Seed: `42`
- Epochs: `10`
- Batch size: `64` (training), `256` (analysis)
- Optimizer: Adam (`lr=1e-3`)
- Loss: CrossEntropyLoss

## Primary Metrics

Source: `metrics/cnn_mnist.csv`

- Best validation accuracy: **98.99%** (epoch 9; ties epoch 10)
- Final validation accuracy: **98.99%**
- Final validation loss: **0.040671**

Test-set evaluation from saved artifacts:
- Overall test accuracy: **99.20%** (`9920/10000`)
- Source: `artifacts/phase3/per_class_accuracy.txt`

## Per-Class Accuracy (Test)

Source: `artifacts/phase3/per_class_accuracy.txt`

| Class | Accuracy | Correct / Total |
| --- | --- | --- |
| 0 | 0.9959 | 976 / 980 |
| 1 | 1.0000 | 1135 / 1135 |
| 2 | 0.9922 | 1024 / 1032 |
| 3 | 0.9941 | 1004 / 1010 |
| 4 | 0.9878 | 970 / 982 |
| 5 | 0.9888 | 882 / 892 |
| 6 | 0.9916 | 950 / 958 |
| 7 | 0.9903 | 1018 / 1028 |
| 8 | 0.9856 | 960 / 974 |
| 9 | 0.9921 | 1001 / 1009 |

Hardest classes in this run: `8`, `4`, `5`.

## Confusion Matrix and Failure Modes

Diagonal (correct predictions per class):
- 0: 976
- 1: 1135
- 2: 1024
- 3: 1004
- 4: 970
- 5: 882
- 6: 950
- 7: 1018
- 8: 960
- 9: 1001

Largest off-diagonal confusions:
- `4 -> 9`: 9
- `9 -> 4`: 5
- `5 -> 3`: 5
- `8 -> 9`: 4
- `7 -> 9`: 4
- `7 -> 2`: 4
- `8 -> 0`: 3
- `6 -> 0`: 3
- `4 -> 6`: 3
- `3 -> 5`: 3

These errors are consistent with handwritten style ambiguity among loop-heavy and
curved digits, not with unstable overall model behavior.

## Artifacts and Evidence

- Metrics log: `metrics/cnn_mnist.csv`
- Best checkpoint: `checkpoints/cnn_mnist_best.pt`
- Last checkpoint: `checkpoints/cnn_mnist_last.pt`
- Confusion matrix: `artifacts/phase3/confusion_matrix.png`
- Misclassified gallery: `artifacts/phase3/misclassified.png`
- Per-class report: `artifacts/phase3/per_class_accuracy.txt`

## Known Limits of This Phase

- Single-symbol digit recognition only.
- No operator classes in this model.
- No end-to-end expression parsing/evaluation in this phase.
