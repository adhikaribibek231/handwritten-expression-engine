# Phase 1 — Data Ingestion and Inspection

## Objective

This phase established whether MNIST was a suitable proxy dataset for the perception layer. Before training any model, the project needed to confirm basic assumptions about image shape, class balance, background noise, centering, and the kinds of ambiguities that would remain even with a strong classifier.

The phase was necessary because later modeling results are only interpretable when the underlying data distribution is understood. If digits were poorly centered, heavily imbalanced, or inconsistent in scale, model errors could easily be misread as architecture problems when they were actually data problems. Phase 1 therefore supports the engineering decision to begin with MNIST and to prioritize spatial models over purely intensity-based baselines.

## Implementation

- `notebooks/01_mnist_exploration.ipynb` loaded MNIST with `torchvision.datasets.MNIST`.
- The notebook worked with the same split structure later used by training scripts: `50,000` training samples, `10,000` validation samples, and the standard `10,000`-image test set.
- `ToTensor()` preprocessing converted images to `float32` tensors with shape `(1, 28, 28)` and normalized pixel values into `[0,1]`.
- The inspection flow covered random sample grids, one sample per digit, mean image per class, and a global pixel-intensity histogram.
- Additional debug artifacts focused on sample-level inspection and a `1` vs `7` pixel-intensity comparison, since that ambiguity was already visible during EDA.
- No model training, checkpoint saving, or metrics logging strategy was introduced in this phase. The deliverables were visual artifacts and dataset-level observations.

## Key Improvements Over Previous Phase

Phase 0 defined the system boundary, but it did not test whether the chosen starting dataset matched that plan. Phase 1 converted the project from architectural intent into data-backed assumptions.

The important change was that later phases no longer needed to guess about data cleanliness or class balance. By the end of this phase, preprocessing and modeling decisions could be grounded in observed properties of the dataset rather than intuition.

## Results

The EDA confirmed that MNIST is clean, normalized, and structurally well-suited for a first handwritten-symbol recognizer. The notebook recorded tensors as `float32` with shape `(1, 28, 28)` and value range `[0.0, 1.0]`, which matches the expectations used later in training scripts.

Class balance was near-uniform across splits rather than perfectly flat. In the training split, counts ranged from `4,546` for digit `5` to `5,613` for digit `1`; validation ranged from `875` to `1,129`; test ranged from `892` to `1,135`. That spread is small enough that class imbalance is not the dominant risk in early classifier behavior.

The visual artifacts showed that digits are consistently centered with low background noise, and the mean-per-class images confirmed that class identity is driven by spatial stroke structure rather than raw intensity totals. The debug comparison around `1` and `7` also showed intrinsic overlap, which is important because it sets a realistic ceiling on perfectly clean separation even when the training loop is correct.

## Failure Analysis

This phase did not yet evaluate model failures, but it did surface future failure modes in the data itself. The clearest ambiguity was `1` versus `7`, where slight hooks or cross-strokes can change interpretation without changing the overall amount of foreground ink very much.

The pixel-intensity histogram also made a second limitation clear: most pixels are background, so intensity statistics alone are not discriminative enough for handwritten recognition. That directly argues against relying on naive flattened representations as a final solution and motivates convolutional feature extraction in later phases.

## Threshold Evaluation

Not applicable in this phase.

## Artifacts Generated

- `notebooks/01_mnist_exploration.ipynb`
- `artifacts/phase1/random_samples.png`
- `artifacts/phase1/one_per_digit.png`
- `artifacts/phase1/mean_per_class.png`
- `artifacts/phase1/pixel_histogram.png`
- `artifacts/phase1/debug/pixel_intensity_histogram_1_vs_7.png`
- `artifacts/phase1/debug/sample_0_digit_9_idx_41905.png`
- `artifacts/phase1/debug/sample_1_digit_4_idx_7296.png`
- `artifacts/phase1/debug/sample_2_digit_7_idx_1639.png`
- `artifacts/phase1/debug/sample_3_digit_2_idx_48598.png`
- `artifacts/phase1/debug/sample_4_digit_9_idx_18024.png`

## Conclusion

This phase is complete. MNIST is a sound starting dataset for the perception subsystem, and the repository now has concrete evidence that the images are clean, centered, and sufficiently balanced for baseline training.

The main uncertainty that remains is model capacity. EDA showed that some ambiguity is intrinsic to the handwriting itself, but it also made clear that spatial structure matters enough that the next phase should begin with a training baseline and then move quickly toward convolutional models.

## Next Phase

Phase 2 should implement the simplest trainable classifier that can validate the full training pipeline. A dense baseline is the right next step because it can confirm that data loading, loss computation, optimization, checkpointing, and validation all work before the project invests in a stronger CNN.
