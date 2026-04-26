# Phase 5 — Inference Preprocessing

## Objective

Phase 5 closed the main deployment gap left after the robust CNN from Phase 4. Training and benchmark evaluation used already-centered, normalized `28x28` MNIST tensors, but real user input arrives as larger grayscale images with uneven lighting, background texture, and arbitrary digit placement. The goal of this phase was therefore to build the preprocessing pipeline that converts raw single-digit images into MNIST-like inputs before classification.

## Implementation

The preprocessing pipeline is implemented in `preprocessing/image_utils.py` and does the following:

1. Load the image in grayscale.
2. Estimate the local background with Gaussian blur and subtract it to isolate dark ink.
3. Apply Otsu thresholding on the foreground-enhanced image instead of a fixed global threshold.
4. Find the digit bounding box and crop to content.
5. Resize while preserving aspect ratio so the longest side becomes `20` pixels.
6. Place the resized digit on a `28x28` canvas and shift by center of mass.
7. Normalize to `[0, 1]` and return a batched tensor for model inference.

This does not mirror training preprocessing step-for-step, because training consumed preprocessed MNIST tensors directly. It does, however, reconstruct the same effective distribution the model was trained on: centered white-on-black digits in a normalized `28x28` frame.

## Debug And Validation Scripts

Two scripts were added for phase-level validation:

- `scripts/debug_preprocessing.py`
- `scripts/test_preprocessing.py`

`debug_preprocessing.py` saves intermediate images for each handwritten sample under `artifacts/phase5/sample_*/`, including `original.png`, `contrast.png`, `thresholded.png`, `cropped.png`, `resized.png`, `centered.png`, and `final_28x28.png`. It also writes `artifacts/phase5/final_28x28_overview.png` for quick visual inspection of the final normalized digits.

`test_preprocessing.py` intentionally stays label-agnostic and prints only prediction/confidence pairs so arbitrary sample images can be checked manually without encoding ground truth into filenames.

Commands used:

```bash
./.venv/bin/python scripts/debug_preprocessing.py
./.venv/bin/python scripts/test_preprocessing.py
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## Automated Tests

The original phase requirement called for unit tests using MNIST samples. That requirement is now explicitly satisfied in `tests/test_image_utils.py`.

The test module covers two groups:

- Real handwritten sample images from `data/sample_digits/`
- Rendered MNIST test-set samples, one per digit, converted into synthetic white-background inference images before preprocessing

For both groups, the tests verify:

- output tensor shape is `(1, 1, 28, 28)`
- output values remain in `[0, 1]`
- output is non-empty
- final center of mass stays close to the middle of the `28x28` canvas

The rendered MNIST path matters because it checks that preprocessing preserves the expected training-style structure even when the input arrives through the raw inference pipeline rather than directly as a dataset tensor.

## Manual Handwritten Validation

Manual validation used the ten real handwritten images under `data/sample_digits/`. The prediction script now prints only the information needed for manual inspection:

```text
sample_0.jpeg -> Pred: 0 | Conf: 0.9995
sample_1.jpeg -> Pred: 1 | Conf: 0.9247
sample_2.jpeg -> Pred: 2 | Conf: 1.0000
sample_3.jpeg -> Pred: 3 | Conf: 0.9998
sample_4.jpeg -> Pred: 4 | Conf: 0.9997
sample_5.jpeg -> Pred: 5 | Conf: 1.0000
sample_6.jpeg -> Pred: 6 | Conf: 0.9945
sample_7.jpeg -> Pred: 7 | Conf: 0.9985
sample_8.jpeg -> Pred: 8 | Conf: 1.0000
sample_9.jpeg -> Pred: 9 | Conf: 0.9851
```

These outputs were manually checked against the handwritten sample set, and the predictions matched the intended digits. The important engineering result is that the final `28x28` stage looks MNIST-like and the trained checkpoint remains confident on real handwritten inputs after preprocessing.

## Main Engineering Decisions

- Fixed-threshold binarization was replaced because it was too fragile under paper lighting gradients.
- `thumbnail()` was removed because inference should explicitly scale the longest side to the MNIST-style inner box rather than only shrinking larger crops.
- Simple geometric centering was replaced with center-of-mass centering because that better matches the visual statistics of MNIST for lopsided digits such as `7` and `9`.
- The inference validation script was kept unlabeled on purpose. For ad hoc testing, the useful question is "what does the model think this image is?" rather than "can the harness infer correctness from the filename?"

## Artifacts Generated

- `preprocessing/image_utils.py`
- `scripts/debug_preprocessing.py`
- `scripts/test_preprocessing.py`
- `tests/test_image_utils.py`
- `artifacts/phase5/final_28x28_overview.png`
- `artifacts/phase5/predictions.csv`
- `artifacts/phase5/predictions.txt`
- `artifacts/phase5/sample_0/` through `artifacts/phase5/sample_9/`

## Conclusion

Phase 5 is complete.

The repository now has a real inference preprocessing pipeline, automated regression coverage that includes MNIST-based unit tests, and manual validation artifacts for real handwritten samples. The main remaining work is no longer single-digit preprocessing but full-expression vision: Phase 6 can now focus on segmentation and bounding-box extraction for multiple symbols in one image.
