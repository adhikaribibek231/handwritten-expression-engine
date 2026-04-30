# Phase 9 — Expression Parsing and Evaluation

## Objective

Phase 8 delivered a complete token sequence from symbol classification — a list like `[12, '+', 3]` ready for symbolic computation. Phase 9 implements the deterministic parser and evaluator that takes this token sequence, validates its correctness, applies operator precedence rules, and computes the final numeric result.

This phase is purely rule-based and mathematical — no images, no neural networks. Correctness is enforced through explicit validation before evaluation: the system either produces a correct numeric answer or raises a deterministic error with a clear reason why.

## Implementation

The parsing and evaluation logic is implemented in `parser/expression_parser.py` as a three-step validation-and-evaluation pipeline.

### Step 1: Token Validation

Function: `validate_tokens(tokens: list) -> None`

Before any computation, the entire token sequence is validated against a strict ruleset. If validation fails, an `ExpressionError` is raised with a diagnostic message. The validation rules are:

**Structural rules:**
- Tokens list must not be empty
- List length must be odd (alternating operands and operators)
- All even-indexed positions (0, 2, 4, ...) must be integers (operands)
- All odd-indexed positions (1, 3, 5, ...) must be strings (operators)

**Operator rules:**
- Only four operators are allowed: `+`, `-`, `×`, `÷`
- Any other symbol raises an `ExpressionError`

**Safety rules:**
- Division by zero is detected before evaluation and raises an explicit error
- This prevents runtime crashes and provides clear user feedback

Example valid tokens: `[2, '+', 3]`, `[12, '+', 3, '×', 4]`  
Example invalid tokens: `[2]` (empty after 2), `[2, '+']` (ends with operator), `[2, '❌', 3]` (unknown operator)

### Step 2: Operator Precedence and Evaluation

Function: `apply_operator(a: int | float, op: str, b: int | float) -> int | float`

Applies a single binary operation. Supports all four operators:
- Addition: `a + b`
- Subtraction: `a - b`
- Multiplication: `a × b`
- Division: `a ÷ b` (with zero-check)

Function: `evaluate_tokens(tokens: list) -> int | float`

Evaluates the validated token sequence in two sequential passes to enforce standard operator precedence (multiplication and division before addition and subtraction):

**Pass 1 — High precedence (`×` and `÷`) left-to-right:**

Scans the token list from left to right. When a `×` or `÷` operator is encountered, it is immediately evaluated with its operands:

```
Input:  [2, '+', 3, '×', 4]
        Position 3 has operator '×'
        Apply:  3 × 4 = 12
        Result: [2, '+', 12]  (continue from position 1)
```

The operator and its two operands are replaced with the computed result, and the scan continues from the left of the replaced section (to handle consecutive multiplications/divisions correctly).

**Pass 2 — Low precedence (`+` and `-`) left-to-right:**

After all high-precedence operators are resolved, the remaining token list contains only `+` and `-`. These are evaluated left-to-right in a second pass:

```
Input:  [2, '+', 12]
        Position 1 has operator '+'
        Apply:  2 + 12 = 14
        Result: [14]  (done)
```

This two-pass approach correctly implements the mathematical rule that multiplication and division are evaluated before addition and subtraction.

**Complex example:**

```
Input:  [2, '+', 3, '×', 4, '-', 5, '÷', 2]

Pass 1 (× and ÷):
  Step 1: Position 3 is '×' → 3 × 4 = 12
  List:  [2, '+', 12, '-', 5, '÷', 2]
  Step 2: Position 5 is '÷' → 5 ÷ 2 = 2.5
  List:  [2, '+', 12, '-', 2.5]

Pass 2 (+ and -):
  Step 1: Position 1 is '+' → 2 + 12 = 14
  List:  [14, '-', 2.5]
  Step 2: Position 1 is '-' → 14 - 2.5 = 11.5
  List:  [11.5]

Result: 11.5
```

### Step 3: Master Function

Function: `parse_and_evaluate(tokens: list) -> int | float`

This is the single public entry point that Phase 10's integration layer calls. It orchestrates validation and evaluation:

1. Calls `validate_tokens(tokens)` — if this raises `ExpressionError`, the exception propagates to the caller
2. If validation passes, calls `evaluate_tokens(tokens)` and returns the result
3. Any arithmetic errors (e.g., division by zero caught during evaluation) raise `ExpressionError`

Return type is `int | float` because:
- All operands from Phase 8 are integers (grouped digits like `12`, `345`)
- Division may produce floats (`5 ÷ 2 = 2.5`)
- Other operations on integers return integers

## Error Handling

All errors are captured and reported through the `ExpressionError` exception class. Examples:

```python
parse_and_evaluate([])
  → ExpressionError: "Empty token sequence"

parse_and_evaluate([2, '+'])
  → ExpressionError: "Invalid token count 2 — expected odd number..."

parse_and_evaluate([2, '❌', 3])
  → ExpressionError: "Unknown operator '❌' at position 1"

parse_and_evaluate([2, '+', 3, '÷', 0])
  → ExpressionError: "Division by zero"
```

Each error message is deterministic and descriptive, enabling:
- User-facing error reporting (Phase 10 displays these messages)
- Logging and debugging
- Automated testing of edge cases

## Design Decisions

1. **No `eval()` function:** The system never uses Python's `eval()` even on sanitized input. Explicit parsing avoids security risks and makes correctness testable.

2. **Strict validation before evaluation:** Token structure is validated before any arithmetic occurs. This prevents cryptic arithmetic errors and provides clear error messages.

3. **Two-pass precedence:** The precedence algorithm is simple, correct, and easy to verify. One pass for high-precedence operators, one pass for low-precedence operators.

4. **Left-to-right evaluation within precedence level:** When multiple operators have the same precedence (e.g., `2 - 3 + 4`), they are evaluated strictly left-to-right. This matches mathematical convention and is deterministic.

5. **Integer operands, float results:** Operands are always integers (from digit grouping in Phase 7/8). Division may produce floats. All other operations on integers return integers.

6. **Explicit exception type:** The custom `ExpressionError` class (inheriting from `Exception`) makes it easy for Phase 10 to distinguish parsing/evaluation errors from other runtime failures.

## Testing and Validation

### Test Coverage

The module includes a `__main__` block with smoke tests:

```python
if __name__ == '__main__':
    # Simple cases
    assert parse_and_evaluate([2, '+', 3]) == 5
    assert parse_and_evaluate([10, '-', 3]) == 7
    assert parse_and_evaluate([4, '×', 5]) == 20
    assert parse_and_evaluate([10, '÷', 2]) == 5.0

    # Precedence
    assert parse_and_evaluate([2, '+', 3, '×', 4]) == 14
    assert parse_and_evaluate([10, '-', 2, '÷', 2]) == 9.0

    # Error cases
    try:
        parse_and_evaluate([2, '+'])
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```

Run via:
```bash
python parser/expression_parser.py
```

### Unit Tests

Comprehensive test suite in `tests/test_expression_parser.py`:

- Valid token sequences with all four operators
- Operator precedence verification
- Left-to-right evaluation within precedence level
- Multi-digit operands
- Error cases (empty list, odd-length mismatch, unknown operators, division by zero)

## Artifacts Generated

- `parser/expression_parser.py` — Core validation and evaluation module
- `parser/__init__.py` — Public API export (`parse_and_evaluate`, `ExpressionError`)
- `tests/test_expression_parser.py` — Comprehensive test suite

## Integration with Phase 8 and Phase 10

**Upstream (Phase 8):** Phase 8's `build_token_sequence()` produces a list like `[12, '+', 3]` that becomes the input to `parse_and_evaluate()`.

**Downstream (Phase 10):** Phase 10's pipeline calls `parse_and_evaluate(tokens)` to compute the final result. On success, the numeric result is returned to the user. On `ExpressionError`, the error message is displayed and logged.

## Conclusion

Phase 9 is complete and implements a robust, deterministic expression evaluator. The system validates all tokens before computation, enforces correct operator precedence, and handles all error conditions with explicit, descriptive messages. No silent failures occur — every invalid input produces a clear diagnostic error.

The implementation is fast (no ML overhead), completely deterministic, and correct by construction through explicit validation and simple, verifiable evaluation logic.

The path is now clear for Phase 10 (End-to-End Integration), which will orchestrate all previous phases into a single continuous system: from raw image input through segmentation, classification, and tokenization, to parsing and final result output.

## Next Phase

Phase 10 should integrate all previous phases into a complete end-to-end system:
- Image input → segmentation → classification → tokenization → parsing → numeric result
- Comprehensive logging at each stage
- Clear error reporting with root cause information
