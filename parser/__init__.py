"""Expression parsing and evaluation module (Phase 9).

This package implements deterministic expression parsing with proper operator
precedence (×÷ before +-) and comprehensive error handling.

Public API:
- parse_and_evaluate(): Validate and evaluate token sequences
- ExpressionError: Custom exception for parsing/evaluation errors

For detailed design and implementation, see docs/results/phase_09.md.
"""

from .expression_parser import parse_and_evaluate, ExpressionError