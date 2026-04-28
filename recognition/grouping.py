"""Token grouping and confidence validation for expression parsing.

After individual symbol recognition, this module handles:
1. Confidence thresholding — reject if digit or operator confidence is low.
2. Digit grouping — merge consecutive digits into multi-digit numbers.
   Example: (1, 0.99), (2, 0.97), (7, 0.95) -> 127
3. Mixed token sequencing — handle both digits and operators.
   Example: (1, 0.99), (2, 0.97), ('+', 0.94), (7, 0.96) -> [12, '+', 7]

Phase 7 assumed all crops were digits. Phase 8 adds operator recognition,
so this module now routes uncertain "digits" to the operator classifier.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

# Confidence thresholds for each symbol type
DIGIT_CONFIDENCE_THRESHOLD = 0.75
OP_CONFIDENCE_THRESHOLD = 0.65


# step 1 - check if any symbol is below its confidence cutoff
def is_low_confidence(results: list[tuple[int | str, float]]) -> bool:
    """Return True if any digit or operator falls below its threshold."""
    for value, conf in results:
        if isinstance(value, int) and conf < DIGIT_CONFIDENCE_THRESHOLD:
            return True
        if isinstance(value, str) and conf < OP_CONFIDENCE_THRESHOLD:
            return True
    return False


# step 2 - merge consecutive digits into a multi-digit number
def group_digits(results: list[tuple[int, float]]) -> list[int]:
    """
    Merge consecutive digit predictions into a full number.

    Input:  [(1, 0.99), (2, 0.97), (7, 0.95)]
    Output: [127]
    """
    if not results:
        return []

    digits = [d for d, _ in results]
    number = int("".join(str(d) for d in digits))
    return [number]


# step 3 - route crops to digit or operator classifier
def classify_all(
    crops: list,
    digit_model: torch.nn.Module,
    operator_model: torch.nn.Module,
    device: torch.device,
) -> list[tuple[int | str, float]]:
    """
    Route each crop to the correct classifier based on digit confidence.

    High digit confidence (>= DIGIT_CONFIDENCE_THRESHOLD) → keep as digit.
    Low digit confidence (< DIGIT_CONFIDENCE_THRESHOLD) → try operator CNN.

    Returns [(value, confidence), ...] where value is int (digit) or str (operator).
    """
    from recognition.digit_recognizer import recognize_digit
    from recognition.operator_recognizer import recognize_operator

    results = []
    for crop in crops:
        digit, conf = recognize_digit(crop, digit_model)
        if conf >= DIGIT_CONFIDENCE_THRESHOLD:
            results.append((digit, conf))
        else:
            op, op_conf = recognize_operator(crop, operator_model)
            results.append((op, op_conf))
    return results


# step 4 - build final token sequence with merged digits
def build_token_sequence(results: list[tuple[int | str, float]]) -> list[int | str]:
    """
    Merge consecutive digits into multi-digit numbers; keep operators separate.

    Input:  [(1, 0.99), (2, 0.97), ('+', 0.94), (7, 0.96)]
    Output: [12, '+', 7]
    """
    tokens = []
    digit_buffer = []

    for value, _ in results:
        if isinstance(value, int):
            digit_buffer.append(value)
        else:
            # hit an operator — flush buffered digits first
            if digit_buffer:
                tokens.append(int("".join(str(d) for d in digit_buffer)))
                digit_buffer = []
            tokens.append(value)

    # flush any remaining digits
    if digit_buffer:
        tokens.append(int("".join(str(d) for d in digit_buffer)))

    return tokens


if __name__ == "__main__":
    # --- Phase 7 tests (digits only) ---
    results = [(2, 0.98), (3, 0.96)]
    print(f"group_digits test: {group_digits(results)}")  # -> [23]

    # --- Phase 8 tests (digits + operators) ---
    from recognition.digit_recognizer import load_model as load_digit_model
    from recognition.operator_recognizer import load_model as load_operator_model
    from vision.segmentation import segment_expression

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digit_model = load_digit_model(device)
    operator_model = load_operator_model(device)

    for path in [
        "data/sample_expressions/sample_0.png",
        "data/sample_expressions/sample_1.png",
        "data/sample_expressions/sample_2.png",
        "data/sample_expressions/sample_3.png",
        "data/sample_expressions/sample_4.png",
        "data/sample_expressions/sample_5.png",
        "data/sample_expressions/sample_6.png",
        "data/sample_expressions/sample_7.png",
    ]:
        print(f"\n{path}")
        _, crops, _ = segment_expression(path)
        results = classify_all(crops, digit_model, operator_model, device)
        print(f"  raw:    {results}")
        if is_low_confidence(results):
            print(f"  ✗ low confidence — ask user to redraw")
        else:
            print(f"  tokens: {build_token_sequence(results)}")