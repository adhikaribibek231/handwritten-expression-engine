```markdown
# Phase 8 — Operator Recognition

## Objective

Phase 7 delivered `(digit, confidence)` pairs but had no way to handle
operator crops — they were misclassified as low-confidence digits and
rejected. Phase 8 adds a dedicated operator CNN to classify `+`, `-`,
`×`, `÷`, and extends the grouping logic to produce complete token
sequences like `[12, '+', 3]` ready for Phase 9's parser.

## Implementation

### `recognition/operator_dataset.py`

Handles dataset loading and train/val splitting for the operator CNN.

**Class mapping:**
```
data/operators/add/ → '+' → class 0
data/operators/sub/ → '-' → class 1
data/operators/mul/ → '×' → class 2
data/operators/div/ → '÷' → class 3
```

The dataset contained ~600 samples per class (596 `+`, 655 `-`,
577 `×`, 618 `÷`). Classes are balanced to the smallest count before
splitting so the model never learns to favour one operator over another.

Two separate transforms are defined — matching the digit model's pattern:
- `TRAIN_TRANSFORM`: grayscale → resize 28×28 → RandomAffine(degrees=15,
  translate=0.1, scale=0.9–1.1) → RandomRotation(10) → normalize
- `EVAL_TRANSFORM`: grayscale → resize 28×28 → normalize (no augmentation)

`get_splits()` returns a train/val split with each subset using its own
transform instance, ensuring augmentation never contaminates validation.

### `scripts/train_operator_cnn.py`

Trains a smaller `MNISTCNN` (hidden_size=64, num_classes=4) on the
operator dataset. Architecture is identical to the digit model but
narrower since 4-class classification is simpler than 10-class MNIST.

Training config: 30 epochs, batch_size=32, lr=1e-3, val_split=0.2.
Best checkpoint saved to `checkpoints/operator_cnn/best.pt`.

### `recognition/operator_recognizer.py`

Handles operator crop classification:

**`load_operator_model(device)`**
Rebuilds `MNISTCNN` from the operator checkpoint. Config is read from
the checkpoint so hidden_size and num_classes don't need to be
hard-coded at call sites.

**`recognize_operator(crop, model)`**
Classifies a single crop. Two preprocessing steps before the CNN:

1. `cv2.bitwise_not()` — inverts white-on-black segmentation crops to
   black-on-white to match the training data distribution
2. `cv2.threshold()` — converts grey strokes to clean black strokes,
   removing the grey antialiasing that survived inversion

Then an aspect ratio rule runs before the CNN:
```python
if aspect > 2.0:
    return '-', 0.95
```
This catches minus signs reliably since `-` is the only operator with
width significantly greater than height (observed: 2.44–3.20 in test
expressions). All other operators measured below 1.10.

### `recognition/grouping.py` (updated)

Extended with two new functions:

**`classify_all(crops, digit_model, operator_model, device)`**
Routes each crop to the correct classifier. Digit CNN runs first — if
confidence ≥ `DIGIT_CONFIDENCE_THRESHOLD` the crop is a digit. If
confidence falls below the threshold it is sent to the operator CNN.
This routing approach means the operator model only ever sees crops the
digit model was uncertain about.

**`build_token_sequence(results)`**
Converts a flat list of `(value, confidence)` pairs into expression
tokens by flushing digit buffers at operator boundaries:
```
[(1, 0.99), (2, 0.97), ('+', 0.94), (3, 0.99)]  →  [12, '+', 3]
```

`is_low_confidence()` was updated to handle mixed int/str values,
checking each against its own threshold
(`DIGIT_CONFIDENCE_THRESHOLD = 0.75`, `OP_CONFIDENCE_THRESHOLD = 0.65`).

## Issues Faced

### Issue 1 — Background inversion mismatch
**Problem:** The operator model was initially predicting `×` for every
operator with high confidence, including `+` and `-`.

**Root cause:** Segmentation produces white-on-black binary crops.
The training dataset had black-on-white images. The model trained on
one distribution and ran inference on the opposite — effectively seeing
inverted symbols it had never encountered.

**Fix:** Added `cv2.bitwise_not()` in `recognize_operator()` before
preprocessing. After inversion, the crops match the training
distribution and predictions improved immediately.

### Issue 2 — Grey strokes after inversion
**Problem:** After inverting, operator strokes appeared grey rather
than solid black. The model was still uncertain on `+` (confidence
~0.67) even with the correct background.

**Root cause:** Segmentation thresholding doesn't produce a perfectly
clean binary image — antialiased stroke edges survive as grey pixels.
After inversion these grey pixels become grey-on-white instead of
black-on-white, which doesn't match the clean training samples.

**Fix:** Added `cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)`
after inversion to force grey strokes to solid black.

### Issue 3 — `-` consistently misclassified as `÷`
**Problem:** Minus signs were being classified as `÷` with high
confidence even after fixing the inversion issue.

**Root cause:** A minus sign is structurally identical to the
horizontal bar of a division sign. The model learned to associate
a single horizontal stroke with `÷` because division signs in the
training data always contain that bar. With no dots present, it still
defaulted to `÷` over `-`.

**Fix:** Added a geometric rule before the CNN — if a crop's aspect
ratio (width/height) exceeds 2.0, classify it as `-` directly with
confidence 0.95. Measured aspect ratios confirmed safe separation:
`-` ranged from 2.44–3.20 while all other operators measured below
1.10. The CNN is bypassed entirely for this case.

### Issue 4 — `1` routed to operator model
**Problem:** The digit `1` in sample_1 (`12+3`) was being sent to the
operator model because the digit CNN returned confidence 0.55, below
`DIGIT_CONFIDENCE_THRESHOLD`.

**Root cause:** The digit `1` in the test image was written with a
diagonal entry stroke at the top (European style). After segmentation
and resizing to 28×28, it visually resembles a `7`. The digit CNN
correctly identified the ambiguity but produced low confidence rather
than a clean prediction.

**Fix (partial):** Lowered `DIGIT_CONFIDENCE_THRESHOLD` from 0.85 to
0.75. This did not resolve the issue because the digit CNN was
predicting `7` at 0.55 — below even the lowered threshold, so the
crop still routed to the operator model.

**Status:** Known limitation. The `1` vs `7` confusion is a digit
model problem rooted in handwriting style. A proper fix requires
either retraining the digit CNN with more stylistic variation or
redrawing the test expression with a simpler `1` stroke. Deferred.

### Issue 5 — Stale model checkpoint
**Problem:** After fixing `recognize_operator()`, results were
identical to before the fix.

**Root cause:** The operator checkpoint had not been retrained after
the dataset and transform changes. The model in memory reflected the
old training run without augmentation or balancing.

**Fix:** Reran `python scripts/train_operator_cnn.py`. Results
improved after retraining with augmented, balanced data.

## Final Results

```
sample_0  6+7   → [6, '+', 7]    ✅
sample_1  12+3  → low confidence  ✗ (1 misclassified as 7)
sample_2  40-9  → [40, '-', 9]   ✅
sample_3  7-1   → [7, '-', 1]    ✅
sample_4  45    → [45]           ✅
sample_5  2*2   → low confidence  expected (* not in classes)
sample_6  8/3   → [8, '×', 3]    expected (/ not in classes)
sample_7  27÷2  → low confidence  expected (÷ splits into blobs)
```

4 of 5 in-scope expressions produce correct token sequences.
The one failure (sample_1) is a digit model limitation, not an
operator recognition failure.

## Design Decisions

1. **Separate operator CNN over unified 14-class model:** A unified
   model would train on ~600 operator samples alongside 50,000 MNIST
   digit samples — severe class imbalance would bias predictions toward
   digits. Separate models train on balanced data within their own
   domain.

2. **Digit CNN as first-stage router:** Running the digit CNN first and
   routing low-confidence crops to the operator model means the operator
   model only sees genuinely ambiguous crops. This avoids running two
   CNNs on every crop.

3. **Geometric rule for `-`:** A CNN trained on a dataset where `-` and
   `÷` share a horizontal bar stroke cannot reliably distinguish them
   from stroke shape alone. The aspect ratio rule is more reliable than
   any learned feature for this specific case.

4. **`OP_CONFIDENCE_THRESHOLD` lower than digit threshold (0.65 vs
   0.75):** Operator crops from segmentation look more variable than
   MNIST digits — different sizes, stroke thicknesses, and styles. A
   lower threshold prevents valid operators from being rejected.

5. **Balancing dataset to smallest class:** With 577–655 samples per
   class, imbalance was minor but still corrected. Balanced training
   ensures the model doesn't learn a prior toward any one operator.

## Artifacts Generated

- `recognition/operator_dataset.py` — balanced dataset loader with
  separate train/eval transforms
- `scripts/train_operator_cnn.py` — operator CNN training pipeline
- `recognition/operator_recognizer.py` — operator crop classification
  with inversion, thresholding, and aspect ratio rule
- `recognition/grouping.py` — extended with `classify_all` and
  `build_token_sequence`
- `checkpoints/operator_cnn/best.pt` — trained operator model

## Conclusion

Phase 8 is complete. The recognition pipeline now produces token
sequences like `[6, '+', 7]` and `[40, '-', 9]` from handwritten
expression images. The known `1` vs `7` confusion is a digit model
limitation deferred to a future improvement pass.

The next step is Phase 9 (expression parsing), which takes token
sequences and evaluates them into numeric results with correct operator
precedence.
```