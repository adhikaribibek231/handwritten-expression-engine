# Phase 02 Results — Baseline Dense Model

## Objective

Validate the MNIST training pipeline with a simple dense baseline before moving to the CNN.

## Experiment Setup

- Script: `scripts/train_baseline.py`
- Model: `models/baseline_dense.py`
- Architecture: `Flatten -> Linear(784,128) -> ReLU -> Linear(128,10)`
- Dataset: MNIST
- Split: train `50,000` / val `10,000` / test `10,000`
- Seed: `42`
- Epochs: `5`
- Batch size: `64`
- Optimizer: Adam (`lr=1e-3`)
- Loss: CrossEntropyLoss

## Metrics

Source: `metrics/baseline_dense.csv`

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| --- | --- | --- | --- | --- |
| 1 | 0.368112 | 0.899100 | 0.219989 | 0.935500 |
| 2 | 0.167892 | 0.951200 | 0.160896 | 0.952100 |
| 3 | 0.118505 | 0.966120 | 0.132716 | 0.959600 |
| 4 | 0.090108 | 0.973960 | 0.116255 | 0.964000 |
| 5 | 0.071147 | 0.979040 | 0.107065 | 0.968300 |

### Summary

- Best validation accuracy: **96.83%** (epoch 5)
- Final validation loss: **0.107065**
- Expected role: baseline sanity check passed

## Interpretation

The baseline reached a strong MNIST sanity-check range and showed stable learning across
all 5 epochs. This validated data loading, transforms, split logic, and training loop
before introducing CNN complexity.

## Artifacts and Evidence

- Metrics: `metrics/baseline_dense.csv`
- Best checkpoint: `checkpoints/baseline_dense_best.pt`
- Last checkpoint: `checkpoints/baseline_dense_last.pt`
- Training script: `scripts/train_baseline.py`

## Known Limits of This Phase

- No spatial inductive bias (flattened input).
- Not intended as final perception system.
- No failure-mode visualization beyond aggregate metrics.
