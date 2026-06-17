# Phase 3 — CNN Perception Model

## Objective

This phase replaced the dense sanity-check model with a CNN that could learn local stroke structure directly from the image. After Phase 2 verified that the repository could train and evaluate a classifier correctly, the next engineering problem was representational: the baseline still collapsed spatial information too early, which limited performance on visually similar digits.

The CNN phase was necessary before any robustness or rejection work because confidence handling only becomes meaningful once the core perception model is strong. Phase 3 therefore supports the decision to treat convolutional feature extraction as the minimum viable architecture for handwritten symbol recognition in Calcinator.

## Implementation

- `models/cnn_mnist.py` defined `MNISTCNN` as `Conv(1,32,3,padding=1) -> ReLU -> MaxPool -> Conv(32,64,3,padding=1) -> ReLU -> MaxPool -> Flatten -> Linear(3136,128) -> ReLU -> Dropout(0.25) -> Linear(128,10)`.
- `scripts/train_cnn.py` trained that model on MNIST using `ToTensor()` preprocessing only, preserving the `(N, 1, 28, 28)` tensor format established earlier.
- The training configuration stayed intentionally comparable to the dense baseline: `SEED=42`, `50,000 / 10,000 / 10,000` split, `10` epochs, batch size `64`, Adam with `lr=1e-3`, and `CrossEntropyLoss`.
- Validation ran every epoch through the shared `evaluate(...)` path, and `metrics/cnn_mnist.csv` was written continuously.
- Checkpoints were stored as `checkpoints/cnn_mnist_best.pt` and `checkpoints/cnn_mnist_last.pt`.
- `scripts/analyze_cnn.py` loaded the best checkpoint and generated a test-set confusion matrix, a per-class accuracy report, and a gallery of misclassified examples under `artifacts/phase3/`.

## Key Improvements Over Previous Phase

Phase 2 proved that the training pipeline worked, but it still treated each image as an unstructured vector. Phase 3 introduced the first model with a spatial inductive bias, allowing the network to learn stroke detectors and translation-tolerant features instead of relying on flattened pixel correlations.

This change matters because the dense baseline's remaining errors were not random. They were concentrated in classes such as `8`, `9`, `4`, and `2`, where topology and local curvature decide the label. The CNN was better because it attacked that exact weakness rather than only adding parameter count.

## Results

The CNN improved best validation accuracy from `96.83%` in Phase 2 to `98.99%` in `metrics/cnn_mnist.csv`, with validation loss reaching `0.0407`. On the test set, `artifacts/phase3/per_class_accuracy.txt` recorded `99.20%` accuracy (`9,920 / 10,000`), reducing test errors from `284` in the dense baseline to `80`.

The improvement was not just aggregate accuracy. Class `1` reached `100.00%` test accuracy, and the hardest classes shifted upward as well: `8` reached `98.56%`, `4` reached `98.78%`, and `5` reached `98.88%`. That confirms the CNN was capturing spatial cues the dense model could not preserve.

The error distribution also became more focused. The largest remaining confusion pairs were `4 -> 9` (`9` cases), `9 -> 4` (`5`), `5 -> 3` (`5`), and smaller pockets involving `8`, `7`, `2`, and `0`. At this stage the classifier was strong enough to be considered the first real perception model rather than a pipeline check.

## Failure Analysis

The misclassified gallery and confusion matrix show that the remaining errors are dominated by handwriting ambiguity rather than broad model instability. Most failures occur when loops are open or closed unusually, when the top bar of a `7` resembles a `2`, or when a `4` acquires a rounded form that resembles a `9`.

Retrospective confidence analysis on the same phase-3 checkpoint, saved later as `metrics/baseline_failure_analysis.csv`, shows why accuracy alone was not sufficient for deployment. The plain CNN made `37` wrong test predictions at confidence `>= 0.90`, and wrong predictions had mean confidence `0.8181`. In other words, the classifier was accurate but still willing to be confidently wrong.

## Threshold Evaluation

Not applicable in this phase. The model always emitted a label, and explicit rejection logic was added only in Phase 4.

## Artifacts Generated

- `models/cnn_mnist.py`
- `scripts/train_cnn.py`
- `scripts/analyze_cnn.py`
- `metrics/cnn_mnist.csv`
- `checkpoints/cnn_mnist_best.pt`
- `checkpoints/cnn_mnist_last.pt`
- `artifacts/phase3/confusion_matrix.png`
- `artifacts/phase3/misclassified.png`
- `artifacts/phase3/per_class_accuracy.txt`

## Conclusion

This phase is complete. The project now has a high-quality digit recognizer with strong test performance and diagnostic artifacts that explain where the remaining errors come from. The core perception path is no longer the primary technical risk.

What this phase did not yet solve is operational safety. The plain CNN still forces a prediction even on ambiguous inputs, which is acceptable for benchmarking but not for a calculator that must avoid silently returning wrong symbols.

## Next Phase

Phase 4 should add robustness and uncertainty handling on top of this CNN baseline. That means training with controlled augmentation, measuring confidence behavior explicitly, and introducing threshold-based rejection so the perception layer can decline uncertain inputs instead of fabricating confident mistakes.
