"""Digit grouping and confidence validation.

After individual digit recognition, this module handles two things:
1. Reject the result if any digit has low confidence.
2. Merge consecutive digit predictions into multi-digit numbers
   (e.g. [1, 2, 3] -> 123).

Right now all crops are assumed to be digits. Once operator recognition
lands in Phase 8, grouping will split runs of digits at operator boundaries.
"""

from __future__ import annotations

CONFIDENCE_THRESHOLD = 0.85


# step 1 - check if any digit is below the confidence cutoff
def is_low_confidence(
    results: list[tuple[int, float]],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> bool:
    """Return True if any digit falls below the confidence threshold."""
    return any(conf < threshold for _, conf in results)


# step 2 - merge consecutive digits into a full number
def group_digits(results: list[tuple[int, float]]) -> list[int]:
    """
    Merge consecutive digit predictions into full numbers.

    Example:
        [(1, 0.99), (2, 0.97), (7, 0.95)]  ->  [127]
        [(1, 0.99), (2, 0.97)]              ->  [12]

    For now this assumes all crops are digits (no operators yet).
    """
    if not results:
        return []

    digits = [d for d, _ in results]
    number = int("".join(str(d) for d in digits))
    return [number]


if __name__ == "__main__":
    # hardcoded test
    results = [(2, 0.98), (3, 0.96)]
    print(group_digits(results))  # -> [23]

    # real results from digit_recognizer
    from recognition.digit_recognizer import load_model, recognize_all
    from vision.segmentation import segment_expression
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    boxes, crops, binary = segment_expression(
        "data/sample_expressions/sample_0.png",
    )
    results = recognize_all(crops, model)
    print("raw results:", results)

    if is_low_confidence(results):
        print("low confidence detected — redraw or check operator presence")
    else:
        print("grouped:", group_digits(results))