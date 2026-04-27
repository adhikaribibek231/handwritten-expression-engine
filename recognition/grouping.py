CONFIDENCE_THRESHOLD = 0.85

def is_low_confidence(results: list[tuple])-> bool:
    """Return true if any digit falls below the confidence threshold."""
    return any(conf <CONFIDENCE_THRESHOLD for _, conf in results)

def group_digits(results: list[tuple]) -> list[int]:
    """
    Merge consecutive digit predictions into full numbers.
    Example:
        [(1, 0.99), (2, 0.97), (7, 0.95)]  →  [127]
        [(1, 0.99), (2, 0.97)]              →  [12]   
    for now this assumes all crops are digits     
    """

    if not results:
        return []
    digits = [d for d, _ in results]

    number = int("".join(str(d) for d in digits))
    return [number]

if __name__ == '__main__':
    # hardcoded test
    results = [(2, 0.98), (3, 0.96)]
    print(group_digits(results))   # → [23]

    # real results from digit_recognizer
    from digit_recognizer import load_model, recognize_all
    from vision.segmentation import segment_expression
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)

    boxes, crops, binary = segment_expression('data/sample_expressions/sample_0.png')
    results = recognize_all(crops, model)
    print("raw results:", results)

    if is_low_confidence(results):
        print("low confidence detected — operator present or bad drawing")
    else:
        print("grouped:", group_digits(results))