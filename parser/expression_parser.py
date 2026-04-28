"""Expression parsing and evaluation.

Takes a validated token sequence from Phase 8 and computes the result.
Never uses eval(). Never fails silently.

Token sequence format:
    [int, str, int, str, int, ...]
    must start and end with int
    operators must alternate with operands
"""

from __future__ import annotations

ALLOWED_OPERATORS = {'+', '-', '×', '÷'}


class ExpressionError(Exception):
    """Raised when a token sequence is invalid or evaluation fails."""
    pass


# ── Step 1 — validate ────────────────────────────────────────────────────────

def validate_tokens(tokens: list) -> None:
    """
    Validate a token sequence before evaluation.
    Raises ExpressionError with a clear message if invalid.

    Rules:
        - must not be empty
        - must start with a number
        - must end with a number
        - operators and numbers must alternate
        - only allowed operators permitted
        - no division by zero
    """
    if not tokens:
        raise ExpressionError("Empty token sequence")

    if len(tokens) % 2 == 0:
        raise ExpressionError(
            f"Invalid token count {len(tokens)} — "
            "expected odd number (operands alternate with operators)"
        )

    for i, token in enumerate(tokens):
        if i % 2 == 0:
            # even positions must be numbers
            if not isinstance(token, int):
                raise ExpressionError(
                    f"Expected number at position {i}, got '{token}'"
                )
        else:
            # odd positions must be operators
            if not isinstance(token, str):
                raise ExpressionError(
                    f"Expected operator at position {i}, got '{token}'"
                )
            if token not in ALLOWED_OPERATORS:
                raise ExpressionError(
                    f"Unknown operator '{token}' at position {i}"
                )

    # check division by zero before evaluation
    for i, token in enumerate(tokens):
        if token == '÷' and tokens[i + 1] == 0:
            raise ExpressionError("Division by zero")


# ── Step 2 — apply precedence ────────────────────────────────────────────────

def apply_operator(a: int | float, op: str, b: int | float) -> int | float:
    """Apply a single binary operator."""
    if op == '+': return a + b
    if op == '-': return a - b
    if op == '×': return a * b
    if op == '÷':
        if b == 0:
            raise ExpressionError("Division by zero")
        return a / b
    raise ExpressionError(f"Unknown operator '{op}'")


def evaluate_tokens(tokens: list) -> int | float:
    """
    Evaluate a validated token sequence with correct precedence.

    Pass 1: resolve × and ÷ left to right
    Pass 2: resolve + and - left to right

    Example:
        [2, '+', 3, '×', 4]
        pass 1 → [2, '+', 12]
        pass 2 → [14]
    """
    # work on a mutable copy
    tokens = list(tokens)

    # pass 1 — multiplication and division
    i = 1
    while i < len(tokens):
        if tokens[i] in ('×', '÷'):
            result = apply_operator(tokens[i - 1], tokens[i], tokens[i + 1])
            tokens[i - 1 : i + 2] = [result]
            # don't advance i — new token is now at i-1
        else:
            i += 2

    # pass 2 — addition and subtraction
    i = 1
    while i < len(tokens):
        result = apply_operator(tokens[i - 1], tokens[i], tokens[i + 1])
        tokens[i - 1 : i + 2] = [result]

    return tokens[0]


# ── Step 3 — master function ─────────────────────────────────────────────────

def parse_and_evaluate(tokens: list) -> int | float:
    """
    Validate and evaluate a token sequence.
    This is the only function Phase 10 needs to call.

    Args:
        tokens: list like [12, '+', 3] from Phase 8

    Returns:
        Numeric result.

    Raises:
        ExpressionError: if tokens are invalid or evaluation fails.
    """
    validate_tokens(tokens)
    return evaluate_tokens(tokens)