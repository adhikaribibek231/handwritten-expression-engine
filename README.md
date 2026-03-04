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
**Current phase:** Phase 4 — Failure analysis and robustness  
**Last updated:** 2026-03-04

### Phase Progress Snapshot

| Phase | Focus | Status | Evidence |
| --- | --- | --- | --- |
| 0 | Project framing + system contract | ✅ Complete | `README.md`, `docs/phases.md` |
| 1 | MNIST data inspection | ✅ Complete | `notebooks/01_mnist_exploration.ipynb`, `artifacts/phase1/` |
| 2 | Baseline dense model | ✅ Complete | `models/baseline_dense.py`, `scripts/train_baseline.py`, `metrics/baseline_dense.csv` |
| 3 | CNN perception model | ✅ Complete | `models/cnn_mnist.py`, `scripts/train_cnn.py`, `metrics/cnn_mnist.csv`, `artifacts/phase3/` |
| 4 | Failure analysis + robustness | 🚧 In Progress | `docs/phases.md` |
| 5-11 | Preprocessing → integration → extensions | ⏳ Planned | `docs/phases.md` |

Detailed phase plan: `docs/phases.md`

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
python scripts/train_cnn.py
python scripts/train_baseline.py
python scripts/analyze_cnn.py
```

Note: `train_cnn.py` downloads MNIST automatically; baseline training expects MNIST
to exist under `data/`.

---

## Reproducibility

- Python: 3.x (project currently developed in local `.venv`)
- Framework: PyTorch (`torch`, `torchvision`)
- Seed: `42` (baseline and CNN scripts)
- Dataset: MNIST
- Metrics logs:
  - `metrics/baseline_dense.csv`
  - `metrics/cnn_mnist.csv`

When adding experiment claims, include script, seed, split, and metric definitions.

---

## Results Summary (High-Level)

### Baseline Dense (Phase 2)

- Best validation accuracy: **96.83%** (`metrics/baseline_dense.csv`)
- Purpose: sanity-check pipeline before CNN work
- Detailed report: `docs/results/phase_02.md`

### CNN (Phase 3)

- Test accuracy: **99.20%** (`artifacts/phase3/per_class_accuracy.txt`)
- Main failure classes: `8`, `4`, `5` are hardest per-class buckets in current run
- Detailed report: `docs/results/phase_03.md`

Artifacts:
- `artifacts/phase3/confusion_matrix.png`
- `artifacts/phase3/misclassified.png`
- `artifacts/phase3/per_class_accuracy.txt`

---

## Roadmap

### Completed

- [x] Phase 0 — Project framing
- [x] Phase 1 — Data ingestion and inspection
- [x] Phase 2 — Baseline dense model
- [x] Phase 3 — CNN digit recognition

### In Progress

- [ ] Phase 4 — Failure analysis and robustness

### Planned

- [ ] Phase 5 — Inference preprocessing
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
- `docs/revision_phase0_2.md` — implementation recap through Phase 2
- `docs/maths_req.md` — math scope for the project
- `docs/results/README.md` — index of detailed phase result reports
- `docs/results/phase_02.md` — detailed Phase 2 metrics and interpretation
- `docs/results/phase_03.md` — detailed Phase 3 metrics and failure analysis
- `artifacts/` — generated figures and debug outputs

---

## Known Limitations (Current State)

- End-to-end calculator pipeline is not integrated yet.
- Current trained models are MNIST digit classifiers (single-symbol scope).
- Operator detection, token grouping, and parser modules are planned phases.

---

## License

No license file is currently defined in this repository.
