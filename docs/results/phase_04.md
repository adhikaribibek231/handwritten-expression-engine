# Phase 4 — Failure Analysis and Robustness

## Objective

Phase 3 delivered a strong CNN classifier, but it still behaved like a benchmark model rather than a safe component in a larger system. For Calcinator, raw top-1 accuracy is not enough: the perception subsystem must either return a symbol that downstream deterministic logic can trust or explicitly decline uncertain inputs. The purpose of Phase 4 was therefore to turn the classifier into a safer decision system.

This phase was necessary before preprocessing, segmentation, and end-to-end expression handling because every downstream stage compounds symbol-level mistakes. A parser cannot recover if the perception layer confidently injects the wrong token, so the engineering decision behind this phase was to improve both robustness under small handwriting variations and calibration of prediction confidence.

## Implementation

- `scripts/train_cnn_robust.py` retrained the same `MNISTCNN` architecture from Phase 3 so the effect of robustness work could be isolated from architectural changes.
- Training data used `RandomAffine` augmentation with `degrees=10`, `translate=(0.1, 0.1)`, and `scale=(0.95, 1.05)`. Validation and test data remained clean and used only `ToTensor()`.
- The phase kept the same core training strategy for comparability: `SEED=42`, `50,000 / 10,000 / 10,000` split, `10` epochs, batch size `64`, Adam at `1e-3`, and `CrossEntropyLoss`.
- Instead of relying on `random_split`, the script built the train and validation splits from a fixed seeded permutation and applied augmentation only to the training subset. That preserved reproducibility while avoiding augmentation leakage into validation.
- Metrics were written each epoch to `metrics/cnn_robust.csv`, and checkpointing saved `checkpoints/cnn_robust_best.pt` and `checkpoints/cnn_robust_last.pt`.
- `scripts/analyze_failures.py` loaded the best robust checkpoint, ran the full test set, computed softmax probabilities, and exported per-sample records to `metrics/robust_failure_analysis.csv`.
- The same analysis pass generated `artifacts/phase4/robust_confusion_matrix.png`, `robust_per_class_accuracy.txt`, `robust_top_confusions.txt`, `robust_failure_summary.txt`, `robust_high_confidence_errors.png`, `robust_low_confidence_correct.png`, and `robust_confidence_hist.png`.
- Baseline comparison artifacts for the unaugmented Phase 3 CNN are also present in the phase directory and corresponding `metrics/baseline_failure_analysis.csv`.
- `scripts/evaluate_thresholds.py` consumed `metrics/robust_failure_analysis.csv`, swept thresholds from `0.50` to `0.98`, wrote `metrics/threshold_evaluation.csv`, and saved `artifacts/phase4/threshold_curve.png`.

## Key Improvements Over Previous Phase

The previous phase produced a high-accuracy CNN, but it had two weaknesses for system use: it was trained on a narrow visual distribution, and it had no mechanism to reject uncertain predictions. Phase 4 addressed both without changing the core network architecture.

The improvement is visible in behavior, not just in top-line accuracy. Test accuracy rose from `99.20%` to `99.30%`, but more importantly mean confidence on wrong predictions dropped from `0.8181` to `0.7082`. High-confidence errors at `>= 0.90` fell from `37` for the plain CNN to `13` for the robust model. The classifier became slightly better at prediction and substantially better at not sounding certain when it was wrong.

## Results

Validation accuracy improved from Phase 3's `98.99%` to `99.07%` at epoch 9 in `metrics/cnn_robust.csv`, and test accuracy increased to `99.30%` (`9,930 / 10,000`) in `artifacts/phase4/robust_per_class_accuracy.txt`. The gain is modest in absolute terms, but it confirms that the added augmentation did not trade away clean-data performance.

The more important result is the shift in error quality. Correct predictions averaged `0.9936` confidence, while wrong predictions averaged `0.7082`. Relative to the plain CNN baseline, high-confidence mistakes were sharply reduced: wrong predictions at `>= 0.95` dropped from `30` to `5`, and even at `>= 0.98` only `3` wrong predictions remained.

Per-class improvements were concentrated in the visually ambiguous buckets that mattered most. Relative to the plain CNN, class accuracy improved by `+0.56` points for `5`, `+0.41` for `4`, `+0.39` for `2`, and `+0.31` for `8`. Small regressions appeared on already-strong classes such as `0`, `1`, `3`, and `9`, which is a reasonable tradeoff for a more conservative model on difficult shapes.

## Failure Analysis

The robust confusion matrix shows that ambiguity has not disappeared; it has become more localized and better calibrated. The largest remaining confusion pair is `7 -> 2` with `5` cases, followed by `9 -> 4`, `8 -> 2`, `4 -> 9`, `3 -> 5`, and `0 -> 6` at `4` each. These are still handwriting-shape failures rather than pipeline failures.

The high-confidence error gallery confirms that the remaining mistakes are dominated by genuinely ambiguous samples: a narrow `1` with a heavy bottom stroke that resembles `2`, open-loop `9` and `4` variants, `5/6` transitions, and deformed `8` shapes that collapse toward `1`, `2`, or `9`. The model is still occasionally wrong with conviction, but the number of such cases is now small enough to manage operationally.

The low-confidence correct gallery is equally important. Some correct symbols score as low as `0.249`, and many sub-`0.60` correct predictions contain broken strokes, faint secondary marks, or unusual proportions. That means confidence now aligns better with difficulty, but rejection thresholds will inevitably discard some valid symbols as the price of avoiding unsafe acceptance.

## Threshold Evaluation

Threshold rejection was introduced because the perception subsystem should not force a token when the evidence is weak. `scripts/evaluate_thresholds.py` tested thresholds at `0.50`, `0.60`, `0.70`, `0.80`, `0.90`, `0.95`, and `0.98` using the per-sample confidence export from `metrics/robust_failure_analysis.csv`.

The resulting tradeoff is clear in `metrics/threshold_evaluation.csv`. At `0.80`, coverage is `98.51%` with `99.73%` accepted accuracy. At `0.90`, coverage drops slightly to `97.70%`, but accepted accuracy rises to `99.87%`, leaving only `13` wrong predictions among `9,770` accepted symbols. At `0.95`, accepted accuracy reaches `99.95%`, but coverage falls to `96.63%`, which means an additional `107` symbols are rejected to eliminate only `8` more accepted mistakes.

A threshold of `0.90` is the best practical operating point for this phase. It removes the majority of unsafe predictions while preserving enough symbol coverage that downstream expression-level acceptance does not collapse under multi-token compounding. In system terms, this phase converts the classifier from a forced-choice predictor into a component that can explicitly ask for a redraw when perception evidence is weak.

## Artifacts Generated

- `scripts/train_cnn_robust.py`
- `scripts/analyze_failures.py`
- `scripts/evaluate_thresholds.py`
- `metrics/cnn_robust.csv`
- `metrics/baseline_failure_analysis.csv`
- `metrics/robust_failure_analysis.csv`
- `metrics/threshold_evaluation.csv`
- `checkpoints/cnn_robust_best.pt`
- `checkpoints/cnn_robust_last.pt`
- `artifacts/phase4/baseline_confusion_matrix.png`
- `artifacts/phase4/baseline_confidence_hist.png`
- `artifacts/phase4/baseline_high_confidence_errors.png`
- `artifacts/phase4/baseline_low_confidence_correct.png`
- `artifacts/phase4/baseline_per_class_accuracy.txt`
- `artifacts/phase4/baseline_top_confusions.txt`
- `artifacts/phase4/baseline_failure_summary.txt`
- `artifacts/phase4/robust_confusion_matrix.png`
- `artifacts/phase4/robust_confidence_hist.png`
- `artifacts/phase4/robust_high_confidence_errors.png`
- `artifacts/phase4/robust_low_confidence_correct.png`
- `artifacts/phase4/robust_per_class_accuracy.txt`
- `artifacts/phase4/robust_top_confusions.txt`
- `artifacts/phase4/robust_failure_summary.txt`
- `artifacts/phase4/threshold_curve.png`

## Conclusion

This phase is complete and marks a meaningful engineering checkpoint. The classifier is now not only accurate but also materially safer: it is more robust to small handwriting variations, less overconfident on wrong predictions, and equipped with a rejection policy that can preserve system integrity.

The major risk that has been de-risked is silent symbol corruption. What remains before end-to-end calculator integration is not basic digit recognition quality but the rest of the perception pipeline: inference preprocessing, segmentation, token grouping, operator recognition, and parser/evaluator integration.

## Next Phase

Phase 5 should build the inference preprocessing pipeline. The current work enables that step because the classifier is now strong enough that remaining deployment risk shifts to distribution matching: thresholding, denoising, aspect-ratio-preserving resize, centering, and normalization must reproduce the MNIST-style input conditions that these models were trained on.
