# Calcinator Architecture

A complete technical overview of the system design and data flow.

---

## System Boundary

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CALCINATOR SYSTEM v1.0                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT: Grayscale handwritten arithmetic expression image         │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ PERCEPTION SUBSYSTEM (ML/CV) — Uncertain output            │   │
│  ├────────────────────────────────────────────────────────────┤   │
│  │                                                            │   │
│  │ Phase 6: Segmentation (Classical CV)                       │   │
│  │   Input: Grayscale image                                  │   │
│  │   Output: Individual cropped symbol images                │   │
│  │   Tech: Otsu thresholding, contour detection, filtering   │   │
│  │                                                            │   │
│  │ Phases 7-8: Classification (Deep Learning)                │   │
│  │   Input: Cropped symbols                                  │   │
│  │   Output: (symbol, confidence) pairs                      │   │
│  │   Tech: CNN for digits, separate CNN for operators       │   │
│  │                                                            │   │
│  └────────────────────────────────────────────────────────────┘   │
│                              │                                     │
│                              ↓ (symbol, confidence) tuples        │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ CONFIDENCE VALIDATION GATE                                 │  │
│  │                                                            │  │
│  │ Rule: Reject if any symbol below threshold               │  │
│  │   - Digits: confidence >= 0.75                           │  │
│  │   - Operators: confidence >= 0.65                        │  │
│  │                                                            │  │
│  │ Consequence: Reject entire expression or proceed         │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                    │
│                              ↓ (all symbols high-confidence)      │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ SYMBOLIC SUBSYSTEM (Rule-Based) — Certain output          │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │                                                            │ │
│  │ Phase 8 (cont'd): Tokenization                             │ │
│  │   Input: (symbol, confidence) tuples                       │ │
│  │   Output: Token sequence [int, str, int, str, ...]         │ │
│  │   Tech: Digit grouping (merge consecutive digits)         │ │
│  │                                                            │ │
│  │ Phase 9: Parsing & Evaluation                              │ │
│  │   Input: Token sequence                                    │ │
│  │   Output: Numeric result (int | float)                    │ │
│  │   Tech: Validation + 2-pass precedence algorithm         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ↓ (numeric result)                  │
│                                                                  │
│  OUTPUT: {"success": true, "result": 13} or error dict         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Core Design Principle: Subsystem Separation

### Perception (ML/CV)
- **Uncertainty**: Produces probabilistic output (confidence scores)
- **Freedom**: Can use heuristics, approximate algorithms
- **Boundary check**: Confidence thresholds gate uncertain output

### Symbolic Reasoning (Rule-Based)
- **Certainty**: Always produces correct answer or explicit error
- **Discipline**: No approximation, all operations deterministic
- **Guarantee**: If accepted by confidence gate, always correct

### Non-Negotiable Interface
- Perception does **NOT** parse or evaluate expressions
- Reasoning does **NOT** inspect raw images
- Interface: Symbol tokens with confidence scores

---

## Module Hierarchy

```
Calcinator (Project Root)
│
├─ api/                          (Phase 10)
│  └─ server.py                  FastAPI HTTP server
│
├─ app/                          (Phase 10)
│  ├─ pipeline.py                Orchestrator (all phases)
│  ├─ main.py                    CLI entry point
│  ├─ logger.py                  Execution logging
│  └─ __init__.py                Public API
│
├─ vision/                       (Phase 6)
│  ├─ segmentation.py            Bounding box detection
│  └─ __init__.py
│
├─ recognition/                  (Phases 7-8)
│  ├─ digit_recognizer.py        Digit CNN inference
│  ├─ operator_recognizer.py     Operator CNN inference
│  ├─ grouping.py                Confidence validation & digit grouping
│  ├─ operator_dataset.py        Operator data loading (training)
│  └─ __init__.py
│
├─ preprocessing/                (Phase 5)
│  ├─ image_utils.py             Preprocessing pipeline
│  └─ __init__.py
│
├─ parser/                       (Phase 9)
│  ├─ expression_parser.py       Token validation & evaluation
│  └─ __init__.py
│
├─ models/                       (Phases 2-4, 8)
│  ├─ baseline_dense.py          Baseline model
│  ├─ cnn_mnist.py               Main digit CNN
│  └─ __init__.py
│
├─ checkpoints/                  Pre-trained weights
│  ├─ cnn_robust/best.pt         Digit model (Phase 4)
│  └─ operator_cnn/best.pt       Operator model (Phase 8)
│
├─ scripts/                      Training and analysis (dev only)
│  ├─ train_baseline.py
│  ├─ train_cnn.py
│  ├─ train_cnn_robust.py
│  ├─ train_operator_cnn.py
│  └─ ...
│
├─ docs/
│  ├─ phases.md                  Phase architecture
│  ├─ maths_req.md               Math prerequisites
│  ├─ results/                   Detailed phase reports
│  │  ├─ phase_00.md through phase_10.md
│  │  └─ README.md               Index
│
├─ README.md                     Main documentation
├─ GETTING_STARTED.md            Quick start guide
└─ ARCHITECTURE.md               This file
```

---

## Data Flow Diagram

### Stage 1: Segmentation (Phase 6)

```
Input Image (512×512 grayscale)
    ↓
Load & Threshold (Otsu)
    ↓ [Binary: white symbols, black background]
Find Contours (OpenCV)
    ↓ [List of contours]
Filter by Size (min_w=5, min_h=5, min_area=150)
    ↓ [Noise removed]
Sort Left-to-Right (x coordinate)
    ↓ [Reading order]
Extract Crops with Padding (pad=4 pixels)
    ↓
Output: [boxes, crops, binary]
```

**Output format:**
- `boxes`: List of (x, y, w, h) tuples
- `crops`: List of numpy arrays, each shape (28, 28) or similar
- `binary`: Binary image for debugging

---

### Stage 2: Classification (Phases 7-8)

```
For each crop:
    ↓
    Preprocess (resize to 28×28, normalize)
    ↓
    Route to Classifier:
      ├─ Digit CNN first (if high confidence → digit)
      └─ Operator CNN if digit confidence too low
    ↓
    Return (value, confidence)
    where value is int (0-9) or str ('+', '-', '×', '÷')
    
Output: [(6, 0.98), ('+', 0.92), (7, 0.97)]
```

**Confidence thresholds:**
- Digit: 0.75
- Operator: 0.65

---

### Stage 3: Tokenization (Phase 8 continued)

```
Input: [(6, 0.98), ('+', 0.92), (7, 0.97)]

Validation: Check all confidence >= threshold
    ✓ Pass → Continue
    ✗ Fail → Return error "Low confidence"

Grouping: Merge consecutive digits into multi-digit numbers
    6 (+) 7  →  No grouping (single digits)
    1 2 (+) 3  →  Group to [12, '+', 3]

Output: [6, '+', 7]
```

---

### Stage 4: Parsing & Evaluation (Phase 9)

```
Input: [6, '+', 7]

Validation:
    ✓ Odd length
    ✓ Alternates operand/operator
    ✓ All operators in {+, -, ×, ÷}
    ✓ No division by zero
    
Evaluation (2-pass precedence):
    Pass 1 (× and ÷ left-to-right): No change → [6, '+', 7]
    Pass 2 (+ and - left-to-right): 6 + 7 = 13 → [13]

Output: 13
```

**Precedence example:**
```
Input: [2, '+', 3, '×', 4]

Pass 1: 3 × 4 = 12
    Result: [2, '+', 12]

Pass 2: 2 + 12 = 14
    Result: [14]

Output: 14 (not 20)
```

---

## Model Architecture

### Digit CNN (Phase 3-4)

```
MNISTCNN(hidden_size=128, num_classes=10)

Input: (batch, 1, 28, 28)
    ↓
Conv2d(1, 32, kernel_size=3)
ReLU
MaxPool2d(2, 2)
    ↓ (batch, 32, 13, 13)
Conv2d(32, 64, kernel_size=3)
ReLU
MaxPool2d(2, 2)
    ↓ (batch, 64, 5, 5)
Flatten → (batch, 1600)
    ↓
Linear(1600, 128)
ReLU
Dropout(0.5)
    ↓
Linear(128, 10)
    ↓ (logits)
Softmax → (batch, 10)

Output: Class probabilities for digits 0-9
```

**Training:**
- Robust version uses data augmentation (rotation, translation, scale)
- Confidence threshold tuned on validation set
- Best checkpoint saved to `checkpoints/cnn_robust/best.pt`

---

### Operator CNN (Phase 8)

```
MNISTCNN(hidden_size=64, num_classes=4)

Same architecture but narrower (64 vs 128) for 4-class classification.

Classes: ['+', '-', '×', '÷']

Preprocessing:
    1. Image inversion (white-on-black → black-on-white)
    2. Threshold to binary (clean strokes)
    3. Aspect ratio heuristic for '-' (if width/height > 2.0, classify as '-')
```

---

## Configuration & Thresholds

### Segmentation Parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `min_w` | 5 | Minimum bounding box width (pixels) |
| `min_h` | 5 | Minimum bounding box height (pixels) |
| `min_area` | 150 | Minimum bounding box area (pixels²) |
| `pad` | 4 | Padding around bounding boxes (pixels) |

---

### Recognition Thresholds

| Threshold | Value | Purpose |
| --- | --- | --- |
| `DIGIT_CONFIDENCE` | 0.85 | Phase 7 digit recognition |
| `DIGIT_CONFIDENCE` (updated) | 0.75 | Phase 8 routing (digit CNN first) |
| `OP_CONFIDENCE` | 0.65 | Phase 8 operator CNN |

---

### Precedence Rules

| Operator | Precedence | Notes |
| --- | --- | --- |
| `×`, `÷` | High | Evaluated first, left-to-right |
| `+`, `-` | Low | Evaluated second, left-to-right |

---

## Error Handling Strategy

### Explicit Error Propagation

```
Every stage can produce an error. No silent failures.

Stage 1 (Segmentation):
    ✗ "No symbols found"
    ✗ "Segmentation failed: [reason]"

Stage 2 (Classification):
    ✗ "Low confidence — please redraw clearly"

Stage 3 (Tokenization):
    (Automatic if Stage 2 passes)

Stage 4 (Parsing):
    ✗ "Invalid expression: [details]"
    ✗ "Division by zero"
    ✗ "Unknown operator: [op]"
```

### Response Format

```json
{
  "success": false,
  "tokens": null,
  "result": null,
  "error": "Low confidence — please redraw clearly",
  "run_id": "20260430_120000_123456"
}
```

All errors returned in response dict with unique run ID for logging.

---

## Logging Architecture

### Execution Trace

Every pipeline run generates `artifacts/runs/<run_id>.json`:

```json
{
  "run_id": "20260430_120000_123456",
  "image": "data/sample_expressions/sample_0.png",
  "stages": [
    {
      "stage": "segmentation",
      "data": {
        "num_symbols": 3,
        "boxes": [[10, 20, 15, 25], ...]
      },
      "failed": false
    },
    {
      "stage": "classification",
      "data": {
        "raw": [["6", 0.981], ["+", 0.923], ["7", 0.967]]
      },
      "failed": false
    },
    ...
  ]
}
```

---

## Deployment Topology

### Single Instance (Development)

```
User Input (image file)
    ↓
python app/main.py
    ↓ (or)
python api/server.py (HTTP)
    ↓
Output (JSON or console)
```

### Containerized (Production)

```
Docker Container
├─ Minimal OS (python:3.13-slim)
├─ Inference dependencies only
├─ Pre-loaded models
└─ FastAPI server
    │
    ├─ GET /health
    ├─ POST /evaluate
    └─ Returns JSON

Deployment targets:
├─ AWS ECS
├─ Google Cloud Run
├─ Azure Container Instances
└─ Kubernetes
```

---

## Performance Profile

| Component | Latency | Throughput |
| --- | --- | --- |
| Segmentation | ~50ms | 1 image/50ms |
| Classification (3 symbols) | ~100ms | 30 symbols/sec |
| Tokenization | <1ms | — |
| Parsing | <1ms | — |
| **End-to-end** | **~150ms** | **6-7 images/sec** |

(Measurements on CPU; GPU would be faster)

---

## Testing Strategy

### Unit Tests

- `tests/test_expression_parser.py` — Parser validation and precedence
- `tests/test_image_utils.py` — Preprocessing pipeline
- `tests/test_segmentation.py` — Bounding box detection
- `tests/test_recognition.py` — Digit/operator classification

### Integration Tests

- `app/main.py` demo mode — Full pipeline on 3 sample images
- API smoke tests in `api/server.py` during startup

### Validation Data

- Sample expressions: `data/sample_expressions/sample_0.png` through `sample_7.png`
- Expected outputs: Encoded in DEMO_SAMPLES in `app/main.py`

---

## Future Extensions (Phase 11+)

| Phase | Feature | Tech |
| --- | --- | --- |
| 11 | Parentheses | Recursive precedence parser |
| 12 | Decimals & negatives | Extended tokenizer |
| 13 | EMNIST (letters) | Reuse CNN architecture |
| 14 | Sequence models | CNN+RNN architecture |
| 15 | User feedback | Correction loop with retraining |

---

## Security Considerations

1. **No user code execution**: Parser never uses `eval()`
2. **Input validation**: All tokens validated before evaluation
3. **Resource limits**: Image size and token count bounded
4. **Error messages**: Diagnostic, non-revealing

---

## Reproducibility

- **Random seed**: 42 (all training scripts)
- **Framework**: PyTorch 2.11.0+
- **Python**: 3.13+
- **OS**: Linux (tested; should work on macOS/Windows)

All results reproducible with same seed and environment.
