# Phase 2 — Baseline Dense Model

## Objective

This phase was intended to prove that the MNIST training pipeline was correct before any architectural sophistication was introduced. After Phase 1 established that the data was clean and mostly centered, the next engineering question was whether the repository could reliably load data, split it reproducibly, optimize a classifier, and save checkpoints without hidden pipeline defects.

The dense baseline was chosen as a diagnostic model rather than a target architecture. If a simple fully connected network could not converge into the expected MNIST range, later CNN work would be untrustworthy because the failure would likely sit in data loading, label handling, tensor shaping, or training-loop logic rather than in model capacity.

## Implementation

- `scripts/train_baseline.py` implemented the complete training loop for the sanity-check classifier.
- `models/baseline_dense.py` defined `MNISTBaseline` with `Flatten -> Linear(784,128) -> ReLU -> Linear(128,10)`.
- Dataset usage followed the Phase 1 contract: MNIST under `data/`, `ToTensor()` preprocessing only, tensors shaped as `(N, 1, 28, 28)`, and pixel values scaled to `[0,1]`.
- The script fixed `SEED=42`, used a `50,000 / 10,000 / 10,000` train/validation/test split, trained for `5` epochs with batch size `64`, and optimized `CrossEntropyLoss` with Adam at `1e-3`.
- Validation was run at the end of every epoch through a shared `evaluate(...)` path so train and validation metrics were computed consistently.
- Metrics were written every epoch to `metrics/baseline_dense.csv` to avoid losing partial progress.
- Checkpointing saved `checkpoints/baseline_dense_best.pt` whenever validation accuracy improved and `checkpoints/baseline_dense_last.pt` at the end of the run.
- The script also evaluated the final model and the best-validation checkpoint on the test set, although those test metrics were printed rather than persisted to a dedicated artifact file.

## Key Improvements Over Previous Phase

Phase 1 established that the data looked usable, but it did not prove that the repository could learn from it. This phase converted static inspection into an executable perception pipeline with deterministic splits, logged metrics, and recoverable model state.

The main improvement was operational rather than architectural: the project moved from observational confidence to measurable training behavior. That made later CNN work a model-comparison problem instead of a pipeline-debugging problem.

## Results

Validation accuracy improved from `93.55%` in epoch 1 to `96.83%` in epoch 5, while validation loss fell to `0.1071` in `metrics/baseline_dense.csv`. That learning curve is the primary success criterion for this phase because it confirms that the data path, optimizer configuration, label handling, and checkpoint logic are all functioning.

Re-evaluating `checkpoints/baseline_dense_best.pt` on the MNIST test set produced `97.16%` accuracy. This is materially above the minimum sanity-check threshold expected for MNIST, but it also leaves a clear performance gap relative to what a spatial model should achieve. The baseline therefore succeeded in its intended role: it validated the pipeline without pretending to be the final perception system.

The class-level behavior also exposed the dense model's structural weakness. Classes `8` and `9` were the hardest test buckets at `94.56%` and `94.15%`, and the largest confusion was `9 -> 4` with `25` cases. The model learned broad class separation, but the flattening step clearly discarded stroke geometry that matters for curved and loop-heavy digits.

## Failure Analysis

The dominant errors came from visually similar classes whose distinction depends on local spatial structure rather than overall intensity mass. The most prominent confusions were `9 -> 4`, `8 -> 3`, `6 -> 0`, `9 -> 3`, `3 -> 5`, and `2 -> 8`. These are exactly the kinds of errors expected when a model sees the image as a 784-dimensional vector instead of a spatial object.

Confidence separation existed but was not yet robust. The mean confidence of correct predictions was `0.9770`, while wrong predictions averaged `0.7047`. That gap shows the network was not randomly uncertain, but it was still confident enough on many incorrect samples that a naive argmax-only deployment would be unsafe.

## Threshold Evaluation

Not applicable in this phase. The baseline was evaluated as a pure classifier, and explicit uncertainty rejection was deferred until Phase 4.

## Artifacts Generated

- `models/baseline_dense.py`
- `scripts/train_baseline.py`
- `metrics/baseline_dense.csv`
- `checkpoints/baseline_dense_best.pt`
- `checkpoints/baseline_dense_last.pt`

## Conclusion

This phase is complete. The repository now has a verified end-to-end training path, reproducible metric logging, and stable checkpoint outputs. The project has therefore de-risked the operational side of model training.

What remains unresolved is representational quality. The dense model is adequate for proving the pipeline, but its error profile shows that flattening the image is the wrong inductive bias for handwritten symbol recognition.

## Next Phase

Phase 3 should replace the dense architecture with a CNN so the model can learn local stroke detectors and translation-tolerant features. The pipeline built here makes that comparison meaningful because any improvement can now be attributed to model design rather than infrastructure instability.
