# Calcinator — Handwritten Calculator

## Status Snapshot (as of 2026-03-04)

- Project stage: active development
- Completed: Phase 0, Phase 1, Phase 2, Phase 3
- In progress next: Phase 4 (failure analysis and robustness)
- Source of phase plan: `docs/phases.md`

## Phase Progress

| Phase | Focus | Status | Evidence |
| --- | --- | --- | --- |
| 0 | Project framing and system contract | Complete | This README (sections below) |
| 1 | MNIST data inspection | Complete | `notebooks/01_mnist_exploration.ipynb`, `artifacts/phase1/` |
| 2 | Baseline dense model | Complete | `models/baseline_dense.py`, `scripts/train_baseline.py`, `metrics/baseline_dense.csv` |
| 3 | CNN perception model | Complete | `models/cnn_mnist.py`, `scripts/train_cnn.py`, `metrics/cnn_mnist.csv`, `artifacts/phase3/` |
| 4 | Failure analysis + robustness | Pending | `docs/phases.md` |
| 5 | Inference preprocessing | Pending | `docs/phases.md` |
| 6 | Digit segmentation | Pending | `docs/phases.md` |
| 7 | Recognition + grouping | Pending | `docs/phases.md` |
| 8 | Operator recognition | Pending | `docs/phases.md` |

## Repro Commands (Through Phase 3)

```bash
python scripts/train_baseline.py
python scripts/train_cnn.py
python scripts/analyze_cnn.py
```

## 1. Project Goal
This project builds a handwritten calculator that converts a single atomic input—
a grayscale image containing a handwritten arithmetic expression—into a single
numeric result or an explicit error.

The system prioritizes correctness over coverage and refuses to compute when uncertain.

## 2. System Overview
The system is composed of two independent subsystems:
1. A perception system that recognizes handwritten symbols probabilistically.
2. A symbolic system that parses and evaluates expressions deterministically.

These systems are strictly separated and communicate only through a fixed symbol
interface. No subsystem is allowed to bypass this boundary.

## 3. Role of Machine Learning
Machine learning is used only for recognizing handwritten digits and operators
from image segments.

ML outputs probabilities, not decisions.

Machine learning is never used for arithmetic, operator precedence,
expression parsing, or error handling.

## 4. Rule-Based Logic
All rule-based components are deterministic: the same input symbols always
produce the same output or the same error, including:
- Grouping digits into numbers
- Operator precedence
- Expression validation
- Arithmetic computation

## 5. Boundary of Uncertainty
Uncertainty exists only during symbol recognition.

If any recognized symbol has confidence below a globally defined confidence threshold,
the system rejects the input and returns an explicit error without attempting
any parsing or arithmetic.

Once symbols are accepted, all further computation is exact.

## 6. Error Philosophy
The system does not guess.
Errors are explicit and categorized, including:
- Low-confidence recognition
- Invalid expressions
- Division by zero
The system never silently corrects, guesses, or auto-fixes invalid input.

## 7. Scope (Version 1)
Supported:
- Digits 0–9
- Operators + - × ÷
- Multi-digit numbers
- Expressions are evaluated using standard arithmetic operator precedence.

Not Supported:
- Parentheses
- Decimals
- Negative numbers
- Scientific notation

## 8. System Boundaries (Non-Negotiable)

- The perception system:
  - Accepts images only
  - Outputs symbols with confidence scores
  - Never performs parsing or arithmetic

- The symbolic system:
  - Accepts symbols only
  - Performs parsing and arithmetic deterministically
  - Never accesses images or probabilities

## Phase 2 (Baseline Dense) Results

Baseline metrics are logged in `metrics/baseline_dense.csv` (5 epochs, seed 42).

- Best validation accuracy: **96.83%** (epoch 5)
- Final validation loss: **0.1071**
- Interpretation: baseline exceeds the expected sanity-check range for MNIST and validates the training/data pipeline before CNN training.

## 9. Phase 3 (CNN) Results and Analysis

The Phase 3 CNN reaches **99.20% test accuracy** on MNIST:

`9920 / 10000 = 0.9920`

### 9.1 Confusion Matrix Summary

The confusion matrix is strongly diagonal, which means most predictions are correct:

| True Digit | Correct Predictions |
| --- | --- |
| 0 | 976 |
| 1 | 1135 |
| 2 | 1024 |
| 3 | 1004 |
| 4 | 970 |
| 5 | 882 |
| 6 | 950 |
| 7 | 1018 |
| 8 | 960 |
| 9 | 1001 |

### 9.2 Main Failure Modes (Not Random)

Largest off-diagonal confusions:

| True | Pred | Count | Likely Visual Reason |
| --- | --- | --- | --- |
| 4 | 9 | 9 | vertical stroke + loop ambiguity |
| 5 | 3 | 5 | similar curved stroke pattern |
| 7 | 2 | 4 | handwritten style variation |
| 7 | 9 | 4 | long curved tail resembles 9 |
| 8 | 5 | 4 | loop-heavy structure overlap |
| 0 | 6 | 2 | loop + tail similarity |

These are typical MNIST confusions and indicate style ambiguity, not unstable behavior.

### 9.3 Misclassified Gallery Interpretation

Representative mistakes from `artifacts/phase3/misclassified.png`:

- `3 -> 5`: strong middle stroke can resemble a handwritten 5
- `4 -> 9`: looped 4 can look like 9
- `6 -> 5`: open-top 6 may resemble 5
- `7 -> 9`: curved top/long tail can push prediction toward 9

### 9.4 What the CNN Learned

The model appears to learn stroke-level visual primitives:

- vertical edges
- loops
- curves
- junctions

When handwriting style makes these structures ambiguous (for example, loop + tail), confusion between `6`, `8`, and `9` increases.

### 9.5 Per-Class Difficulty

Hardest classes from `artifacts/phase3/per_class_accuracy.txt`:

| Digit | Accuracy |
| --- | --- |
| 8 | 98.56% |
| 4 | 98.78% |
| 5 | 98.88% |

Easiest class:

- `1`: **100.00%** (`1135/1135`)

Interpretation: loop- and curve-dominant digits vary more in handwriting; digit `1` has minimal structural ambiguity.

## 10. Phase 3 Deliverables Status

Phase 3 deliverables are complete:

- `models/cnn_mnist.py`
- `scripts/train_cnn.py`
- checkpoints in `checkpoints/`
- metrics log in `metrics/cnn_mnist.csv`
- confusion matrix in `artifacts/phase3/confusion_matrix.png`
- per-class accuracy in `artifacts/phase3/per_class_accuracy.txt`
- misclassified gallery in `artifacts/phase3/misclassified.png`

## 11. Next Phase Direction

Likely Phase 4 focus areas:

- multi-digit recognition
- operator classification
- expression parsing

This transitions the project from single-symbol digit recognition to a full calculator vision pipeline.
