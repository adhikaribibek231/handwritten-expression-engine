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
**Latest completed phase:** Phase 10 — End-to-end integration  
**Next phase:** Phase 11 — Extensions (parentheses, decimals, EMNIST, etc.)  
**Last updated:** 2026-04-29

### Phase Progress Snapshot

| Phase | Focus | Status | Evidence |
| --- | --- | --- | --- |
| 0 | Project framing + system contract | ✅ Complete | `README.md`, `docs/phases.md`, `docs/results/phase_00.md` |
| 1 | MNIST data inspection | ✅ Complete | `notebooks/01_mnist_exploration.ipynb`, `artifacts/phase1/`, `docs/results/phase_01.md` |
| 2 | Baseline dense model | ✅ Complete | `models/baseline_dense.py`, `scripts/train_baseline.py`, `metrics/baseline_dense.csv`, `docs/results/phase_02.md` |
| 3 | CNN perception model | ✅ Complete | `models/cnn_mnist.py`, `scripts/train_cnn.py`, `metrics/cnn_mnist.csv`, `artifacts/phase3/`, `docs/results/phase_03.md` |
| 4 | Failure analysis + robustness | ✅ Complete | `scripts/train_cnn_robust.py`, `metrics/cnn_robust.csv`, `artifacts/phase4/`, `docs/results/phase_04.md` |
| 5 | Inference preprocessing | ✅ Complete | `preprocessing/image_utils.py`, `scripts/debug_preprocessing.py`, `scripts/test_preprocessing.py`, `tests/test_image_utils.py`, `docs/results/phase_05.md` |
| 6 | Digit segmentation | ✅ Complete | `vision/segmentation.py`, `scripts/debug_segmentation.py`, `artifacts/phase6/`, `docs/results/phase_06.md` |
| 7 | Recognition & grouping | ✅ Complete | `recognition/digit_recognizer.py`, `recognition/grouping.py`, `docs/results/phase_07.md` |
| 8 | Operator recognition | ✅ Complete | `recognition/operator_recognizer.py`, `scripts/train_operator_cnn.py`, `docs/results/phase_08.md` |
| 9 | Expression parsing & evaluation | ✅ Complete | `parser/expression_parser.py`, `tests/test_expression_parser.py`, `docs/results/phase_09.md` |
| 10 | End-to-end integration | ✅ Complete | `app/pipeline.py`, `app/main.py`, `app/logger.py`, `docs/results/phase_10.md` |
| 11+ | Extensions | ⏳ Planned | `docs/phases.md` |

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
├─ api/                   # FastAPI server (production)
├─ app/                   # End-to-end pipeline & CLI (Phase 10)
├─ artifacts/             # Plots and visual outputs by phase [dev only]
├─ checkpoints/           # Saved model weights [needed for inference]
├─ data/                  # MNIST data root [dev/training only]
├─ docs/                  # Phase plans and project notes [reference]
├─ metrics/               # CSV logs for training/evaluation [dev only]
├─ models/                # Model definitions [needed for inference]
├─ notebooks/             # EDA and exploratory analysis [dev only]
├─ parser/                # Expression parsing & evaluation (Phase 9)
├─ preprocessing/         # Inference preprocessing pipeline
├─ recognition/           # Digit & operator recognition (Phase 7-8)
├─ scripts/               # Train / analysis scripts [dev/training only]
├─ tests/                 # Automated tests [dev only]
├─ vision/                # Segmentation & classical CV (Phase 6)
└─ README.md
```

---

## Deployment: Bare Minimum Dependencies

The project is **complete and ready for production**. To run the API or inference, you only need these dependencies:

```
torch>=2.11.0
torchvision>=0.26.0
opencv-python>=4.13.0.92
numpy>=2.4.3
fastapi>=0.136.1
uvicorn>=0.46.0
python-multipart>=0.0.27
```

**Files needed for inference:**
- `app/` — End-to-end pipeline
- `api/` — FastAPI server
- `checkpoints/` — Pre-trained model weights
- `models/` — Model architectures
- `recognition/` — Classifier modules
- `vision/` — Segmentation module
- `preprocessing/` — Image preprocessing
- `parser/` — Expression evaluation

**Files NOT needed for inference** (development/training only):
- `scripts/` — Training scripts
- `notebooks/` — EDA and analysis
- `data/` — MNIST dataset
- `metrics/` — Training logs
- `artifacts/` — Visualizations
- `tests/` — Unit tests
- `docs/` — Documentation

---

## Quick Start

### For Production / API

#### 1) Install minimal dependencies

```bash
pip install torch torchvision opencv-python numpy fastapi uvicorn python-multipart
```

#### 2) Run the API server

```bash
python api/server.py
```

The server listens on `http://localhost:8000` by default.

**Endpoints:**
- `GET /health` — Health check
- `POST /evaluate` — Send an image, get result

Example request:
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "accept: application/json" \
  -F "file=@data/sample_expressions/sample_0.png"
```

#### 3) Run demo (no server)

```bash
python app/main.py
```

Runs the full pipeline on pre-selected sample images and displays results.

#### 4) Process single image

```bash
python app/main.py data/sample_expressions/sample_0.png
```

---

### For Development / Training

#### 1) Install all dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project + all dev dependencies
uv sync
```

This creates a `.venv` and installs everything from `pyproject.toml` + `uv.lock`.

#### 2) Run training scripts

```bash
uv run python scripts/train_baseline.py
uv run python scripts/train_cnn.py
uv run python scripts/analyze_cnn.py
uv run python scripts/train_cnn_robust.py
uv run python scripts/analyze_failures.py
uv run python scripts/evaluate_thresholds.py
```

Note: `train_cnn.py` downloads MNIST automatically; baseline training expects MNIST
to exist under `data/`.

---

## Deployment Configuration

### Running the API Server

The project includes a FastAPI server for production deployment:

```bash
pip install torch torchvision opencv-python numpy fastapi uvicorn python-multipart
python api/server.py
```

**Server capabilities:**
- Accepts handwritten expression images via HTTP POST
- Processes through the complete pipeline (segmentation → classification → parsing)
- Returns JSON with success status, tokens, and numeric result
- CORS-enabled for cross-origin requests
- Health check endpoint

**Docker deployment** (optional):
You can containerize the API for cloud deployment (AWS ECS, GCP Cloud Run, etc.). The minimal Dockerfile would install only the inference dependencies listed above.

### Model Checkpoints Required

Ensure these checkpoint files are present:
- `checkpoints/cnn_robust/best.pt` — Digit recognition model
- `checkpoints/operator_cnn/best.pt` — Operator recognition model

These are included in the repository and do not need retraining for inference.

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

### Digit Segmentation (Phase 6)

- Seven-step pipeline implemented in `vision/segmentation.py`: load → threshold → find contours → filter → sort → crop
- Uses Otsu's adaptive thresholding for robust binarization across variable lighting
- Segments multi-symbol expressions into individual cropped symbol images via bounding-box detection
- Debug script `scripts/debug_segmentation.py` generates visual overlays and per-sample artifacts
- Validated on 8 handwritten expression samples; correctly handles clean, spaced input
- Known limitations documented: touching symbols, disconnected components (e.g., ÷), with mitigations
- Detailed report: `docs/results/phase_06.md`

### Recognition & Grouping (Phase 7)

- Digit recognition via robust CNN checkpoint on segmented crops
- Preprocessing reuses Phase 5's `preprocess_crop_for_inference()` to match MNIST distribution
- Confidence threshold: **0.85** — reject if any single digit falls below
- Consecutive digits grouped into multi-digit numbers (e.g. `[1, 2, 3] → 123`)
- All-digit assumption by design; operators deferred to Phase 8
- Detailed report: `docs/results/phase_07.md`

### Operator Recognition (Phase 8)

- Dedicated CNN for 4-class operator classification: `+ - × ÷`
- Dataset: ~600 samples per operator with balanced training/validation splits
- Architecture: MNISTCNN (hidden_size=64, num_classes=4)
- Preprocessing: Image inversion + threshold to match training distribution
- Aspect ratio heuristic: `-` detected via width/height ratio (>2.0)
- Routing logic: Digit CNN first, operator CNN for low-confidence crops
- Confidence thresholds: 0.75 for digits, 0.65 for operators
- Detailed report: `docs/results/phase_08.md`

### Expression Parsing & Evaluation (Phase 9)

- Three-stage pipeline: validation → precedence evaluation → master function
- Validation: Checks token structure, operator validity, division by zero
- Precedence: Two-pass algorithm (×÷ left-to-right, then +- left-to-right)
- Error handling: Deterministic ExpressionError with diagnostic messages
- No `eval()` — all parsing explicit and verifiable
- Detailed report: `docs/results/phase_09.md`

### End-to-End Integration (Phase 10)

- Complete pipeline: Image → segmentation → classification → tokenization → parsing → result
- FastAPI server with `/health` and `/evaluate` endpoints
- CORS-enabled for cross-origin requests
- Structured JSON responses with success/error/tokens/result
- Comprehensive logging to `artifacts/runs/<run_id>.json` for debugging
- Demo suite validates 3 sample expressions (100% pass rate)
- Detailed report: `docs/results/phase_10.md`

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

### V1 System — Complete ✅

All core phases are complete and production-ready:

- [x] Phase 0 — Project framing and system contract
- [x] Phase 1 — Data ingestion and inspection
- [x] Phase 2 — Baseline dense model
- [x] Phase 3 — CNN digit recognition
- [x] Phase 4 — Failure analysis and robustness
- [x] Phase 5 — Inference preprocessing
- [x] Phase 6 — Digit segmentation (classical CV)
- [x] Phase 7 — Digit recognition and grouping
- [x] Phase 8 — Operator recognition and routing
- [x] Phase 9 — Expression parsing and evaluation
- [x] Phase 10 — End-to-end integration with API

### V1+ Extensions — Planned

- [ ] Phase 11 — Parentheses and operator precedence extensions
- [ ] Phase 12 — Decimal and negative number support
- [ ] Phase 13 — EMNIST support (letters for algebraic expressions)
- [ ] Phase 14 — Sequence models (CNN+RNN) for improved accuracy
- [ ] Phase 15 — User feedback and correction loops

Full plan: `docs/phases.md`

---

## Documentation

### Getting Started
- **[GETTING_STARTED.md](GETTING_STARTED.md)** — Quick start guide for running the system (< 5 minutes)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Complete technical architecture and data flow diagrams

### Architecture & Planning
- `docs/phases.md` — Phase-by-phase goals, deliverables, and design decisions
- `docs/maths_req.md` — Mathematical scope and learning prerequisites
- `docs/currentfolderstructure.md` — Repository structure visualization

### Phase Reports (Detailed Technical Documentation)
- `docs/results/README.md` — Index of all phase reports
- `docs/results/phase_00.md` — Project framing and system contract
- `docs/results/phase_01.md` — MNIST data inspection and visualization
- `docs/results/phase_02.md` — Baseline dense model performance
- `docs/results/phase_03.md` — CNN digit recognition and failure analysis
- `docs/results/phase_04.md` — Robustness through data augmentation and confidence thresholds
- `docs/results/phase_05.md` — Inference preprocessing pipeline
- `docs/results/phase_06.md` — Classical computer vision segmentation
- `docs/results/phase_07.md` — Digit recognition and multi-digit grouping
- `docs/results/phase_08.md` — Operator recognition with routing logic
- `docs/results/phase_09.md` — Expression parsing with operator precedence
- `docs/results/phase_10.md` — End-to-end pipeline and FastAPI integration

### Generated Artifacts
- `artifacts/` — Visualizations, confusion matrices, and debug outputs by phase

---

## Known Limitations (By Design)

### V1 Scope (Intentional)

These features are explicitly deferred to Phase 11+:

- **Parentheses**: Not supported (Phase 11)
- **Decimals**: Only integers supported (Phase 12)
- **Negative numbers**: Unary negation not implemented (Phase 12)
- **Letters/EMNIST**: Only digits and four operators (Phase 13)

### Segmentation Edge Cases

**Current limitations** (documented in Phase 6):
- **Touching symbols**: When symbols share pixels, they may be detected as one blob. Workaround: ensure clear spacing in input.
- **Disconnected components**: Symbols like `÷` with separate parts (dots + line) may split. Workaround: use higher `min_area` threshold or substitute `/`.
- **Extreme sizes**: Very large or small symbols may not pass size filters. Workaround: tune `min_w`, `min_h`, `min_area` parameters.

### Operator Recognition

- **Aspect ratio heuristic for minus**: Relies on `-` being significantly wider than tall (observed: 2.44-3.20 ratio). Custom handwriting styles may not conform.
- **Pre-processing assumption**: White-on-black image format assumed; images with different contrast may need preprocessing adjustment.

---

## Performance Metrics (V1 Complete System)

### End-to-End Accuracy

On sample validation set (3 handwritten expressions):
- Demo suite: **3/3 correct** ✅
- Pass rate: **100%** on sample_0.png (6+7), sample_2.png (40-9), sample_3.png (7-1)

### Component Performance

| Component | Metric | Value |
| --- | --- | --- |
| Digit CNN | Test accuracy | 99.30% |
| Digit CNN | Operating threshold | 0.90 (97.70% coverage, 99.87% accepted accuracy) |
| Operator CNN | Architecture | MNISTCNN (hidden_size=64, num_classes=4) |
| Segmentation | Robustness | Otsu adaptive thresholding |
| Parser | Precedence | Standard (×÷ before +-) |
| Parser | Error handling | Deterministic with diagnostic messages |

---

## Contributing

See `docs/phases.md` for phase-level architecture. Each phase is self-contained with clear deliverables.

For local development:
```bash
uv sync
uv run python scripts/train_baseline.py  # Example training script
```

---

## License

No license file is currently defined in this repository. This is a portfolio/educational project.
