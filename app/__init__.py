"""Calcinator application layer — end-to-end pipeline orchestration (Phase 10).

This package integrates all previous phases (1-9) into a single continuous system
for recognizing handwritten arithmetic expressions and computing their results.

Key modules:
- pipeline: Core orchestration of all stages from image to result
- main: CLI entry point and demo interface
- logger: Structured logging for debugging and monitoring

For detailed design, see docs/results/phase_10.md.
"""

from .pipeline import run, load_models