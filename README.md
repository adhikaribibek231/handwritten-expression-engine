# Calcinator — Handwritten Expression Calculator

> Probabilistic symbol recognition + deterministic expression evaluation.

---

## Overview

Calcinator is a phased project for building an image-to-result handwritten calculator.
The input is a grayscale handwritten expression image, and the output is either a numeric
result or an explicit error when confidence/syntax is insufficient.

The design keeps a hard boundary between:
- Perception (ML/CV, uncertain)
- Reasoning (rule-based parsing/evaluation, deterministic)

This boundary is the main engineering contract for the repository.

---

## Current Status

**Project stage:** active development  
**Latest completed phase:** Phase 5 — Inference preprocessing  
**Next phase:** Phase 6 — Digit segmentation  
**Last updated:** 2026-04-26

### Phase Progress Snapshot

| Phase | Focus | Status | Evidence |
| --- | --- | --- | --- |
| 0 | Project framing + system contract | ✅ Complete | `README.md`, `docs/phases.md`, `docs/results/phase_00.md` |
| 1 | MNIST data inspection | ✅ Complete | `notebooks/01_mnist_exploration.ipynb`, `artifacts/phase1/`, `docs/results/phase_01.md` |
| 2 | Baseline dense model | ✅ Complete | `models/baseline_dense.py`, `scripts/train_baseline.py`, `metrics/baseline_dense.csv`, `docs/results/phase_02.md` |
| 3 | CNN perception model | ✅ Complete | `models/cnn_mnist.py`, `scripts/train_cnn.py`, `metrics/cnn_mnist.csv`, `artifacts/phase3/`, `docs/results/phase_03.md` |
| 4 | Failure analysis + robustness | ✅ Complete | `scripts/train_cnn_robust.py`, `metrics/cnn_robust.csv`, `artifacts/phase4/`, `docs/results/phase_04.md` |
| 5 | Inference preprocessing | ✅ Complete | `preprocessing/image_utils.py`, `scripts/debug_preprocessing.py`, `scripts/test_preprocessing.py`, `tests/test_image_utils.py`, `docs/results/phase_05.md` |
| 6-11 | Segmentation → integration → extensions | ⏳ Planned | `docs/phases.md` |

Detailed phase plan: `docs/phases.md`  
Detailed phase reports: `docs/results/README.md`

---

## Problem Definition (System Contract)

### Goal

Convert a handwritten arithmetic expression image into a correct numeric answer, or
reject with an explicit error if recognition confidence or syntax validity is not met.

### Input

- Grayscale image containing handwritten digits/operators.

### Output

- Numeric result, or explicit error (for example low confidence, invalid expression,
  division by zero).

### Version 1 Scope

**Supported**
- Digits `0-9`
- Operators `+ - × ÷`
- Multi-digit numbers
- Standard arithmetic precedence

**Not supported (v1)**
- Parentheses
- Decimals
- Negative numbers
- Scientific notation

---

## Core Architecture

### 1) Perception Subsystem (ML/CV)

Responsible for:
- Symbol recognition from images
- Confidence estimation
- Segmentation/classification artifacts

Output interface:
- Sequence of recognized symbols + confidence scores

### 2) Symbolic Subsystem (Rule-Based)

Responsible for:
- Token grouping
- Syntax validation
- Parsing
- Deterministic arithmetic evaluation

Output:
- Final result or explicit deterministic error

### Boundary (Non-Negotiable)

- ML does **not** perform arithmetic or parsing.
- Rule-based logic does **not** inspect raw images.
- Interface between subsystems is symbol tokens with confidence.

---

## Design Principles

- Correctness over guesswork
- Explicit errors over silent failure
- Probabilistic perception, deterministic computation
- Reproducible experiments
- Incremental phase-based development

---

## Repository Structure

```text
calcinator/
├─ artifacts/             # Plots and visual outputs by phase
├─ checkpoints/           # Saved model weights
├─ data/                  # MNIST data root
├─ docs/                  # Phase plans and project notes
├─ metrics/               # CSV logs for training/evaluation
├─ models/                # Model definitions
├─ notebooks/             # EDA and exploratory analysis
├─ scripts/               # Train / analysis scripts
└─ README.md
```

---

## Quick Start

### 1) Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 2) Install dependencies

```bash
pip install torch torchvision numpy matplotlib
```

### 3) Run training and analysis

```bash
python scripts/train_baseline.py
python scripts/train_cnn.py
python scripts/analyze_cnn.py
python scripts/train_cnn_robust.py
python scripts/analyze_failures.py
python scripts/evaluate_thresholds.py
```

Note: `train_cnn.py` downloads MNIST automatically; baseline training expects MNIST
to exist under `data/`.

---

## Reproducibility

- Python: 3.x (project currently developed in local `.venv`)
- Framework: PyTorch (`torch`, `torchvision`)
- Seed: `42` (baseline, CNN, and robust CNN scripts)
- Dataset: MNIST
- Metrics logs:
  - `metrics/baseline_dense.csv`
  - `metrics/cnn_mnist.csv`
  - `metrics/cnn_robust.csv`
  - `metrics/baseline_failure_analysis.csv`
  - `metrics/robust_failure_analysis.csv`
  - `metrics/threshold_evaluation.csv`

When adding experiment claims, include script, seed, split, and metric definitions.

---

## Results Summary (High-Level)

### Baseline Dense (Phase 2)

- Best validation accuracy: **96.83%** (`metrics/baseline_dense.csv`)
- Best-checkpoint test accuracy: **97.16%** (`checkpoints/baseline_dense_best.pt`, reevaluated)
- Purpose: sanity-check pipeline before CNN work
- Detailed report: `docs/results/phase_02.md`

### CNN (Phase 3)

- Best validation accuracy: **98.99%** (`metrics/cnn_mnist.csv`)
- Test accuracy: **99.20%** (`artifacts/phase3/per_class_accuracy.txt`)
- Main failure classes: `8`, `4`, `5` are hardest per-class buckets in current run
- Detailed report: `docs/results/phase_03.md`

### Robust CNN + Rejection (Phase 4)

- Best validation accuracy: **99.07%** (`metrics/cnn_robust.csv`)
- Test accuracy: **99.30%** (`artifacts/phase4/robust_per_class_accuracy.txt`)
- Mean confidence on wrong predictions reduced from **0.8181** (plain CNN) to **0.7082** (`artifacts/phase4/baseline_failure_summary.txt`, `artifacts/phase4/robust_failure_summary.txt`)
- Recommended operating threshold: **0.90**, giving **97.70%** coverage with **99.87%** accepted accuracy (`metrics/threshold_evaluation.csv`)
- Detailed report: `docs/results/phase_04.md`

### Inference Preprocessing (Phase 5)

- Implemented in `preprocessing/image_utils.py`
- Debug artifacts written to `artifacts/phase5/`
- `scripts/test_preprocessing.py` now runs arbitrary sample images and prints only `Pred` and `Conf` for manual inspection
- Automated regression coverage includes rendered MNIST test-set samples plus the handwritten sample set in `tests/test_image_utils.py`
- Detailed report: `docs/results/phase_05.md`

Result artifacts:
- `artifacts/phase3/confusion_matrix.png`
- `artifacts/phase3/misclassified.png`
- `artifacts/phase3/per_class_accuracy.txt`
- `artifacts/phase4/robust_confusion_matrix.png`
- `artifacts/phase4/robust_high_confidence_errors.png`
- `artifacts/phase4/robust_low_confidence_correct.png`
- `artifacts/phase4/threshold_curve.png`

---

## Roadmap

### Completed

- [x] Phase 0 — Project framing
- [x] Phase 1 — Data ingestion and inspection
- [x] Phase 2 — Baseline dense model
- [x] Phase 3 — CNN digit recognition
- [x] Phase 4 — Failure analysis and robustness
- [x] Phase 5 — Inference preprocessing

### Planned

- [ ] Phase 6 — Digit segmentation
- [ ] Phase 7 — Recognition and grouping
- [ ] Phase 8 — Operator recognition
- [ ] Phase 9 — Expression parsing and evaluation
- [ ] Phase 10 — End-to-end integration
- [ ] Phase 11 — Extensions

Full plan: `docs/phases.md`

---

## Documentation

- `docs/phases.md` — phase-by-phase goals and deliverables
- `docs/maths_req.md` — math scope for the project
- `docs/results/README.md` — index of detailed phase result reports
- `docs/results/phase_00.md` — Phase 0 architecture and project framing report
- `docs/results/phase_01.md` — Phase 1 dataset inspection report
- `docs/results/phase_02.md` — detailed Phase 2 metrics and interpretation
- `docs/results/phase_03.md` — detailed Phase 3 metrics and failure analysis
- `docs/results/phase_04.md` — detailed Phase 4 robustness and threshold report
- `docs/results/phase_05.md` — detailed Phase 5 preprocessing and validation report
- `artifacts/` — generated figures and debug outputs

---

## Known Limitations (Current State)

- End-to-end calculator pipeline is not integrated yet.
- Current trained models operate on single MNIST-like digit crops rather than full expression images.
- Segmentation, operator recognition, token grouping, parser modules, and full end-to-end integration are still planned phases.
- Confidence rejection is defined for the digit classifier, but expression-level acceptance logic is not yet integrated.

---

## License

No license file is currently defined in this repository.
