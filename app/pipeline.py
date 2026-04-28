"""Core pipeline. Takes an image path, returns a result."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from vision.segmentation import segment_expression
from recognition.grouping import classify_all, build_token_sequence, is_low_confidence
from recognition.digit_recognizer import load_model as load_digit_model
from recognition.operator_recognizer import load_model as load_operator_model
from parser.expression_parser import parse_and_evaluate, ExpressionError
from app.logger import log_run, make_run_id


def load_models(device: torch.device) -> tuple:
    digit_model    = load_digit_model(device)
    operator_model = load_operator_model(device)
    return digit_model, operator_model


def run(
    image_path: str,
    digit_model,
    operator_model,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """
    Run the full pipeline on one image.

    Returns a result dict:
        {
            'success': bool,
            'tokens':  list or None,
            'result':  int | float | None,
            'error':   str | None,
            'run_id':  str,
        }
    """
    run_id = make_run_id()

    def log(stage, data, failed=False):
        if verbose:
            print(f"  [{stage}] {data}")
        log_run(run_id, image_path, stage, data, failed=failed)

    log('input', {'image': image_path})

    # ── Stage 1 — segmentation ───────────────────────────────
    try:
        boxes, crops, binary = segment_expression(image_path)
        log('segmentation', {'num_symbols': len(crops), 'boxes': boxes})
    except Exception as e:
        log('segmentation', {'error': str(e)}, failed=True)
        return {'success': False, 'tokens': None, 'result': None,
                'error': f'Segmentation failed: {e}', 'run_id': run_id}

    if not crops:
        log('segmentation', {'error': 'no symbols found'}, failed=True)
        return {'success': False, 'tokens': None, 'result': None,
                'error': 'No symbols found in image', 'run_id': run_id}

    # ── Stage 2 — classification ─────────────────────────────
    results = classify_all(crops, digit_model, operator_model, device)
    log('classification', {'raw': [(str(v), round(c, 3)) for v, c in results]})

    if is_low_confidence(results):
        log('classification', {'error': 'low confidence'}, failed=True)
        return {'success': False, 'tokens': None, 'result': None,
                'error': 'Low confidence — please redraw clearly', 'run_id': run_id}

    # ── Stage 3 — tokenization ───────────────────────────────
    tokens = build_token_sequence(results)
    log('tokenization', {'tokens': tokens})

    # ── Stage 4 — parsing and evaluation ────────────────────
    try:
        result = parse_and_evaluate(tokens)
        log('evaluation', {'result': result})
        return {'success': True, 'tokens': tokens, 'result': result,
                'error': None, 'run_id': run_id}
    except ExpressionError as e:
        log('evaluation', {'error': str(e)}, failed=True)
        return {'success': False, 'tokens': tokens, 'result': None,
                'error': f'Invalid expression: {e}', 'run_id': run_id}