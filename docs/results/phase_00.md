# Phase 0 — Project Framing

## Objective

This phase defined the system contract for Calcinator before any model training began. The project is not a generic OCR demo; it is an image-to-result calculator with two fundamentally different subsystems: an uncertain perception layer and a deterministic arithmetic layer. The purpose of Phase 0 was to keep that boundary explicit from the start.

That framing was necessary because later phases would otherwise mix responsibilities. If the ML subsystem were allowed to absorb parsing or arithmetic behavior, model evaluation would become ambiguous and system failures would be hard to localize. The Phase 0 decision was therefore architectural: ML would output symbols and confidence, while rule-based logic would own grouping, syntax validation, parsing, and numeric evaluation.

## Implementation

- `README.md` was established as the top-level engineering contract for the repository.
- `docs/phases.md` defined the delivery sequence from project framing through end-to-end integration.
- The version-1 problem boundary was fixed to digits `0-9`, operators `+ - × ÷`, multi-digit numbers, and deterministic arithmetic precedence.
- Explicit non-goals were recorded for v1: no parentheses, decimals, negative numbers, or scientific notation.
- The repository structure was organized around phase-based experimentation: `models/`, `scripts/`, `metrics/`, `artifacts/`, `checkpoints/`, and `docs/`.
- No dataset loading, preprocessing pipeline, training run, validation loop, checkpoint strategy, or metric logger was introduced in this phase by design. The output of Phase 0 was a stable interface definition rather than executable ML.

## Key Improvements Over Previous Phase

This was the starting phase, so there was no prior implementation to improve. The key change was from an idea to an explicit systems contract with clear subsystem ownership and versioned scope.

## Results

The main result of this phase was architectural clarity rather than a numerical metric. The repository now states a non-negotiable boundary: ML handles perception only, and rule-based logic handles symbolic reasoning only. That decision materially reduces later integration risk because it defines what every downstream model must produce: symbol predictions plus confidence scores.

Phase 0 also fixed the immediate product scope. By limiting v1 to handwritten digits and four arithmetic operators, the project established a tractable first milestone that can be validated before more complex notation is introduced.

## Failure Analysis

No model existed yet, so there were no statistical failures to analyze. The remaining risk after this phase was conceptual rather than empirical: the architecture was clear, but it had not yet been tested against real data or trained models.

## Threshold Evaluation

Not applicable in this phase.

## Artifacts Generated

- `README.md`
- `docs/phases.md`

## Conclusion

This phase is complete. The project now has a defensible system boundary, an explicit v1 scope, and a sequencing plan for implementation. That de-risks later work by preventing architectural drift between probabilistic perception and deterministic computation.

What remains is evidence. The next phases must validate that MNIST is a suitable proxy dataset, that the training pipeline works, and that the perception subsystem can produce reliable symbol predictions before any parser is introduced.

## Next Phase

Phase 1 should inspect MNIST in detail. That work matters because preprocessing and modeling decisions depend on understanding centering, stroke thickness, pixel intensity distribution, and intrinsic digit ambiguity before a classifier is trained.
