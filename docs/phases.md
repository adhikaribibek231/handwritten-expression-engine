# Phase 0 — Project framing (do this first, 1–2 hours)

**Goal:** Prevent architectural confusion later.

### Explicit assumptions

* Input: grayscale handwritten digits and operators
* Output: a single numeric result
* Scope (v1):

  * Digits: 0–9
  * Operators: `+ - × ÷`
  * No parentheses initially
  * Left-to-right evaluation or standard precedence (decide now)

### Deliverables

* A one-page README answering:

  * What is ML?
  * What is rule-based?
  * Where does probability end and certainty begin?

This separation will save you weeks later.

---

# Phase 1 — Data ingestion & inspection (MNIST)

**Goal:** Understand the data distribution, not just load it.

### Concrete steps

1. Load MNIST
2. Split into train / val / test
3. Normalize pixels → `[0,1]`
4. Explicitly shape tensors for your framework

   * PyTorch: `(N, 1, 28, 28)`
   * TensorFlow/Keras: `(N, 28, 28, 1)`

### Mandatory inspections

* Plot:

  * Random samples
  * One sample per digit
* Compute:

  * Mean image per class
  * Pixel intensity histogram

### Questions you must answer

* Are digits centered?
* How thick are strokes?
* How much background noise exists?

### Deliverables

* `notebooks/01_mnist_exploration.ipynb`
* Saved plots (commit them)

If you skip this, later preprocessing bugs will look “mysterious.”

---

# Phase 2 — Baseline model (sanity check)

**Goal:** Prove the pipeline works before sophistication.

### Model

* Flatten → Dense(128) → ReLU → Dense(10) → Softmax

### Why this matters

If this fails to reach ~92–94%, your data pipeline is broken.

### Metrics to log

* Training loss
* Validation loss
* Accuracy

### Deliverables

* `models/baseline_dense.py`
* Training script
* Logged metrics

This model is disposable. Treat it as a diagnostic tool.

---

# Phase 3 — CNN model (real perception system)

**Goal:** Learn spatial feature extraction.

### Current implementation architecture

```
Conv(3×3, 32) → ReLU → MaxPool
Conv(3×3, 64) → ReLU → MaxPool
Flatten
Dense(128) → ReLU → Dropout
Dense(10) → logits
```

### Key concepts you should internalize

* Convolutions = stroke detectors
* Pooling = translation tolerance
* Softmax = uncertainty modeling

### Evaluation beyond accuracy

* Confusion matrix
* Per-class accuracy
* Misclassified image gallery

### Deliverables

* `models/cnn_mnist.py`
* Saved weights
* Confusion matrix image

You are not done until you understand *why* errors occur.

---

# Phase 4 — Failure analysis & robustness

**Goal:** Make the model usable, not impressive.

### Analyze failures

* Identify visually ambiguous digits
* Measure confidence scores

### Introduce robustness

* Data augmentation:

  * Slight rotations
  * Translations
  * Thickness variation

### Introduce uncertainty handling

* Reject predictions if `max(prob) < threshold`
* Output: `"uncertain"`

### Deliverables

* Augmented training script
* Confidence threshold logic

This is where ML stops hallucinating.

---

# Phase 5 — Inference preprocessing (critical engineering phase)

**Goal:** Match training distribution at inference.

### Preprocessing pipeline

1. Convert canvas image to grayscale
2. Threshold (adaptive > global)
3. Remove noise
4. Resize while preserving aspect ratio
5. Center digit in 28×28
6. Normalize

### Rule (non-negotiable)

> Inference preprocessing **must mirror** training preprocessing.

### Deliverables

* `preprocessing/image_utils.py`
* Unit tests using MNIST samples

Most OCR systems fail here, not in the model.

---

# Phase 6 — Digit segmentation (vision, not ML)

**Goal:** Turn one image into multiple digit images.

### Steps

1. Threshold full expression image
2. Find contours
3. Filter by size/aspect ratio
4. Compute bounding boxes
5. Sort left → right
6. Crop and preprocess each box

### Edge cases

* Touching digits
* Uneven spacing
* Small noise blobs

### Deliverables

* `vision/segmentation.py`
* Visual debug overlays (bounding boxes drawn)

This phase teaches classical computer vision discipline.

---

# Phase 7 — Recognition & grouping

**Goal:** Convert symbols into numbers.

### Pipeline

* For each cropped digit:

  * Run CNN
  * Store `(digit, confidence)`
* Group consecutive digits:

  * `[1,2,3] → 123`

### Validation

* Reject if any digit is low confidence
* Ask user to redraw

### Deliverables

* `recognition/digit_recognizer.py`
* `recognition/grouping.py`

This is where probability meets determinism.

---

# Phase 8 — Operator recognition (two paths)

### Option A: Rule-based (simpler)

* Separate CNN trained on operators
* Classes: `+ - × ÷`

### Option B: Unified classifier

* Single CNN with 14 classes (digits + operators)

**Recommendation:** Start with Option A.

### Deliverables

* Operator dataset
* Operator model

---

# Phase 9 — Expression parsing & evaluation

**Goal:** Pure symbolic correctness.

### Steps

1. Token sequence: `[123, '+', 45]`
2. Validate syntax
3. Apply precedence
4. Compute result

### Rules

* Never trust raw input
* No direct `eval` without sanitization

### Deliverables

* `parser/expression_parser.py`
* Test cases for edge inputs

This part should never fail silently.

---

# Phase 10 — End-to-end integration

**Goal:** One continuous system.

### Full flow

```
Canvas
→ Image
→ Segmentation
→ Classification
→ Tokenization
→ Parsing
→ Evaluation
→ Result
```

### Instrument everything

* Log intermediate outputs
* Save failure cases

### Deliverables

* `app/main.py`
* End-to-end demo

---

# Phase 11 — Extensions (only after v1 works)

* EMNIST (letters)
* Parentheses
* Sequence models (CNN + RNN)
* Transformer OCR
* User correction feedback loop

---
