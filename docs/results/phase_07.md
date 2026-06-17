# Phase 7 — Recognition and Grouping

## Objective

Phase 6 delivered individually cropped symbol images from a handwritten expression. Phase 7 takes those crops and converts them into actual numbers. Each crop is run through the robust CNN from Phase 4 (via the preprocessing pipeline from Phase 5), producing a `(digit, confidence)` pair. Consecutive digit predictions are then grouped into multi-digit numbers (e.g. `[1, 2, 3] → 123`).

This is the phase where probability meets determinism: the CNN's probabilistic output is validated against a confidence threshold before being handed to the rule-based grouping logic.

## Implementation

Two modules were added under `recognition/`:

### `recognition/digit_recognizer.py`

Handles model loading and per-crop classification in three steps:

**Step 1: Load Model**
Function: `load_model(device)`

Rebuilds the `MNISTCNN` from the Phase 4 checkpoint (`checkpoints/cnn_robust_best.pt`). The checkpoint stores both weights and the hyper-parameter config, so architecture details don't need to be hard-coded. The model is placed in eval mode immediately after loading.

**Step 2: Classify Single Crop**
Function: `recognize_digit(crop, model)`

Takes one cropped symbol image (numpy array) and returns `(predicted_digit, confidence)`. Internally it:
1. Preprocesses the crop via `preprocess_crop_for_inference()` from Phase 5 (resize → center on 28×28 → normalize)
2. Runs a forward pass through the CNN with no gradient tracking
3. Applies softmax to get per-class probabilities
4. Returns the argmax class and its probability

**Step 3: Classify All Crops**
Function: `recognize_all(crops, model)`

Runs `recognize_digit` on every crop in left-to-right order and returns the full list of `(digit, confidence)` tuples.

### `recognition/grouping.py`

Handles the transition from perception output to symbolic tokens:

**Step 1: Confidence Validation**
Function: `is_low_confidence(results, threshold=0.85)`

Returns `True` if any digit in the result list falls below the confidence threshold. A single low-confidence crop is enough to flag the whole expression because one wrong digit changes the entire numeric value.

**Step 2: Digit Grouping**
Function: `group_digits(results)`

Merges consecutive digit predictions into multi-digit numbers. Currently all crops are assumed to be digits, so the entire list collapses into a single number. Once Phase 8 adds operator recognition, this function will split runs at operator boundaries to produce multiple numbers.

### `recognition/__init__.py`

Exports the public API: `load_model`, `recognize_digit`, `recognize_all`, `group_digits`, and `is_low_confidence`.

## Design Decisions

1. **Confidence threshold at 0.85:** Matches the operating threshold established in Phase 4's threshold evaluation. Conservative enough to catch genuine misclassifications without rejecting too many valid predictions.

2. **Per-digit rejection, not per-expression averaging:** A single bad digit corrupts the number (e.g. `13` vs `18`), so the strictest policy is correct — reject if *any* digit is below threshold.

3. **Preprocessing reuse:** Crops are preprocessed using `preprocess_crop_for_inference()` from Phase 5 rather than a separate path. This ensures the model sees the same distribution it was trained on.

4. **All-digit assumption:** Phase 7 intentionally treats every crop as a digit. Operators will be misclassified as low-confidence digits and correctly rejected by the threshold check. This is by design — operator support arrives in Phase 8.

5. **Grouping returns a list:** Even though we currently produce a single number, `group_digits` returns `list[int]` so the interface is ready for Phase 8 when multiple numbers separated by operators will be present.

## Validation

Both modules include `__main__` smoke tests that:
- Load a crop from Phase 6 artifacts and classify it
- Run the full segmentation → recognition → grouping pipeline on a sample expression
- Print raw predictions and grouped output

These can be run directly:
```bash
./.venv/bin/python recognition/digit_recognizer.py
./.venv/bin/python recognition/grouping.py
```

## Artifacts Generated

- `recognition/digit_recognizer.py` — model loading and per-crop classification
- `recognition/grouping.py` — confidence validation and digit grouping
- `recognition/__init__.py` — public API exports

## Conclusion

Phase 7 is complete. The recognition module correctly classifies cropped digits using the trained CNN and groups them into multi-digit numbers. Confidence-based rejection catches unreliable predictions before they reach downstream symbolic processing.

The next step is Phase 8 (operator recognition), which will add `+ - × ÷` classification and extend the grouping logic to produce token sequences like `[123, '+', 45]` ready for expression parsing.
