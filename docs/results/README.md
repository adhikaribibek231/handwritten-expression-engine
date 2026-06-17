# Results Index

This folder contains phase-level technical reports derived from the repository's
source code, metrics, checkpoints, and generated artifacts.

## Available Reports

- [Phase 00 — Project Framing](phase_00.md)
- [Phase 01 — Data Ingestion and Inspection](phase_01.md)
- [Phase 02 — Baseline Dense Model](phase_02.md)
- [Phase 03 — CNN Perception Model](phase_03.md)
- [Phase 04 — Failure Analysis and Robustness](phase_04.md)
- [Phase 05 — Inference Preprocessing](phase_05.md)
- [Phase 06 — Digit Segmentation](phase_06.md)
- [Phase 07 — Digit Recognition and Grouping](phase_07.md)
- [Phase 08 — Operator Recognition and Grouping](phase_08.md)
- [Phase 09 — Expression Parsing and Evaluation](phase_09.md)
- [Phase 10 — End-to-End Integration](phase_10.md)

## Notes

- Phases 02-10 reference measured results from files under `metrics/`,
  `checkpoints/`, and `artifacts/`.
- Phase 01 is based on `notebooks/01_mnist_exploration.ipynb` and the saved plots
  under `artifacts/phase1/`.
- Phase 09 documents the expression parsing and evaluation logic from `parser/expression_parser.py`.
- Phase 10 documents the end-to-end pipeline orchestration from `app/pipeline.py` and `app/main.py`.
- Later phases should continue the same naming pattern: `phase_XX.md`.
