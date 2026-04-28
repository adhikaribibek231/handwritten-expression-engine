# run this from project root
import sys, torch
sys.path.insert(0, '.')

from vision.segmentation import segment_expression
from recognition.grouping import classify_all, build_token_sequence, is_low_confidence
from recognition.digit_recognizer import load_model as load_digit_model
from recognition.operator_recognizer import load_model as load_operator_model
from parser.expression_parser import parse_and_evaluate, ExpressionError

device         = torch.device('cpu')
digit_model    = load_digit_model(device)
operator_model = load_operator_model(device)

samples = [
    'data/sample_expressions/sample_0.png',  # 6+7  → 13
    'data/sample_expressions/sample_2.png',  # 40-9 → 31
    'data/sample_expressions/sample_3.png',  # 7-1  → 6
]

for path in samples:
    _, crops, _ = segment_expression(path)
    results     = classify_all(crops, digit_model, operator_model, device)
    tokens      = build_token_sequence(results)
    try:
        answer = parse_and_evaluate(tokens)
        print(f"{path}  tokens={tokens}  result={answer}")
    except ExpressionError as e:
        print(f"{path}  ERROR: {e}")
