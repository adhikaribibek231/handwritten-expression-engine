"""Expression parsing and evaluation (Phase 9).

Takes a validated token sequence from Phase 8 (recognition and grouping) and
computes the final numeric result using explicit validation and standard
operator precedence rules.

Never uses eval(). Never fails silently. All errors are caught and reported
through the ExpressionError exception with diagnostic messages.

Token sequence format:
    [int, str, int, str, int, ...]
    
    Examples:
    - [6, '+', 7]  → 13
    - [12, '+', 3, '×', 4]  → 24 (not 60, due to precedence)
    - [10, '÷', 2, '-', 3]  → 2.0

    Must start and end with int (operand)
    Must alternate: operand, operator, operand, operator, ...
    Operators must be in ALLOWED_OPERATORS
    Division by zero is caught before evaluation

Phase 9 consists of three stages:
1. Validation: Check token structure, operator validity, division by zero
2. Evaluation: Apply precedence rules (×,÷ before +,-) left-to-right
3. Master function: Orchestrate validation + evaluation

For detailed design rationale and testing, see docs/results/phase_09.md.
"""

from __future__ import annotations

ALLOWED_OPERATORS = {'+', '-', '×', '÷'}


class ExpressionError(Exception):
    """Raised when a token sequence is invalid or evaluation fails.
    
    All errors from Phase 9 are ExpressionError instances with clear,
    diagnostic messages suitable for user display or logging.
    
    Examples:
        ExpressionError("Empty token sequence")
        ExpressionError("Expected number at position 0, got '+'")
        ExpressionError("Division by zero")
    """
    pass


# ── Step 1 — validate ────────────────────────────────────────────────────────

def validate_tokens(tokens: list) -> None:
    """Validate a token sequence before evaluation.
    
    Checks all structural, operator, and safety rules. Raises ExpressionError
    with a clear message if any check fails. This is called before any
    computation to provide early, actionable error feedback.
    
    Validation rules:
    
    **Structural:**
    - Token list must not be empty
    - Length must be odd (alternating operand-operator-operand)
    - Even positions (0, 2, 4, ...) must be int (operands)
    - Odd positions (1, 3, 5, ...) must be str (operators)
    
    **Operator validity:**
    - Only '+', '-', '×', '÷' are allowed
    - Unknown operators raise ExpressionError
    
    **Safety (pre-evaluation checks):**
    - Division by zero is detected early
    - Prevents cryptic arithmetic errors later
    
    Args:
        tokens: List to validate, typically [int, str, int, str, ...]
    
    Raises:
        ExpressionError: If any validation rule fails
        
    Examples:
        validate_tokens([2, '+', 3])  # ✓ Valid
        validate_tokens([2, '+'])     # ✗ Even length
        validate_tokens([2, '+', 0, '÷', 5])  # ✗ 0 ÷ 5
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
    """Apply a single binary operator.
    
    Computes a op b for one of the four allowed operators. This is a helper
    function used by evaluate_tokens during precedence resolution.
    
    Args:
        a: Left operand (int or float)
        op: Operator ('+', '-', '×', or '÷')
        b: Right operand (int or float)
    
    Returns:
        Result of a op b (int for +,-,× on ints; float for ÷)
        
    Raises:
        ExpressionError: If op is unknown or b=0 for ÷
        
    Examples:
        apply_operator(2, '+', 3)  → 5
        apply_operator(10, '÷', 2)  → 5.0
    """
    if op == '+': return a + b
    if op == '-': return a - b
    if op == '×': return a * b
    if op == '÷':
        if b == 0:
            raise ExpressionError("Division by zero")
        return a / b
    raise ExpressionError(f"Unknown operator '{op}'")


def evaluate_tokens(tokens: list) -> int | float:
    """Evaluate a validated token sequence with correct precedence.
    
    Implements standard arithmetic precedence: multiply and divide before
    add and subtract, left-to-right within each precedence level.
    
    Algorithm (two-pass):
    
    **Pass 1 — High precedence (× and ÷) left-to-right:**
    Scans the token list and immediately evaluates any × or ÷ operator with
    its operands. The operator and operands are replaced by their result.
    
        [2, '+', 3, '×', 4]
        Sees × at position 3
        Computes: 3 × 4 = 12
        Result:   [2, '+', 12]
    
    **Pass 2 — Low precedence (+ and -) left-to-right:**
    After all high-precedence operators are resolved, any remaining operators
    are + or -, which are evaluated left-to-right.
    
        [2, '+', 12]
        Computes: 2 + 12 = 14
        Result:   [14]
    
    Complex example:
        Input:  [2, '+', 3, '×', 4, '-', 5, '÷', 2]
        
        Pass 1: Resolve × and ÷
          3 × 4 = 12  →  [2, '+', 12, '-', 5, '÷', 2]
          5 ÷ 2 = 2.5 →  [2, '+', 12, '-', 2.5]
        
        Pass 2: Resolve + and -
          2 + 12 = 14 →  [14, '-', 2.5]
          14 - 2.5 = 11.5 → [11.5]
        
        Final result: 11.5
    
    Args:
        tokens: Validated token list from validate_tokens()
    
    Returns:
        Numeric result (int or float)
        
    Raises:
        ExpressionError: If apply_operator fails (should not happen if
                        validate_tokens was called first)
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
    """Validate and evaluate a token sequence in one call.
    
    This is the only function Phase 10's pipeline needs to call. It orchestrates
    both validation and evaluation, ensuring tokens are correct before any
    computation occurs.
    
    The function is the public API for Phase 9. Callers should catch
    ExpressionError for error handling.
    
    Args:
        tokens: List like [12, '+', 3] from Phase 8 (recognition/grouping)
                Must alternate: int, str, int, str, ...
    
    Returns:
        Numeric result (int or float).
        
        Int for +, -, × on integer operands.
        Float for ÷ or when ÷ appears in the expression.
    
    Raises:
        ExpressionError: If tokens fail any validation check or evaluation
                        error occurs (e.g., division by zero).
    
    Examples:
        parse_and_evaluate([6, '+', 7])  → 13
        parse_and_evaluate([10, '-', 3])  → 7
        parse_and_evaluate([2, '+', 3, '×', 4])  → 14 (not 20)
        parse_and_evaluate([10, '÷', 2])  → 5.0
        
        Error cases:
        parse_and_evaluate([2, '+'])
            → ExpressionError: "Invalid token count 2 — expected odd..."
        parse_and_evaluate([2, '+', 0, '÷', 5])
            → ExpressionError: "Division by zero"
    """
    validate_tokens(tokens)
    return evaluate_tokens(tokens)