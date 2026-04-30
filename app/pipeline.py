"""End-to-end pipeline orchestration (Phase 10).

This module orchestrates all previous phases (1-9) into a single continuous
system for recognizing handwritten arithmetic expressions and computing results.

The pipeline stages are:

1. Segmentation (Phase 6):    Image → crops
2. Classification (Phase 7-8): Crops → (value, confidence) pairs
3. Tokenization (Phase 8):    Pairs → tokens [12, '+', 3]
4. Parsing & Evaluation (Phase 9): Tokens → numeric result

All intermediate outputs are logged to artifacts/runs/ for debugging and
analytics. Clear error messages are returned at any failure point.

For detailed design rationale and data flows, see docs/results/phase_10.md.
"""

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
    """Load digit and operator CNN models from checkpoints.
    
    Both models are loaded in evaluation mode (no gradient tracking) from
    their respective checkpoints under checkpoints/:
    - Digit model: checkpoints/cnn_robust/best.pt
    - Operator model: checkpoints/operator_cnn/best.pt
    
    Args:
        device: torch.device (typically torch.device('cpu') or torch.device('cuda'))
    
    Returns:
        tuple: (digit_model, operator_model) both in eval mode
    """
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
    """Run the complete end-to-end pipeline on a single handwritten expression image.
    
    Orchestrates all stages: segmentation → classification → tokenization →
    parsing → evaluation. Returns a structured result dict regardless of
    success or failure.
    
    **Pipeline stages:**
    
    1. **Segmentation**: Loads image, thresholds to binary, finds symbol contours,
       filters/sorts bounding boxes, extracts individual crops.
       
    2. **Classification**: Routes each crop to digit or operator CNN, producing
       (value, confidence) pairs.
       
    3. **Confidence Validation**: Checks all values meet thresholds
       (0.75 for digits, 0.65 for operators). Rejects entire expression if
       any value is too low.
       
    4. **Tokenization**: Groups consecutive digits into multi-digit numbers,
       produces token sequence [12, '+', 3].
       
    5. **Parsing & Evaluation**: Validates token syntax, applies operator
       precedence, computes numeric result.
    
    **Logging**: Each stage is logged (console + disk) with stage-specific data.
    
    Args:
        image_path: Path to input image file (grayscale handwritten expression)
        digit_model: Loaded digit CNN model (from load_models)
        operator_model: Loaded operator CNN model (from load_models)
        device: torch.device for model inference
        verbose: If True, print stage progress to console
    
    Returns:
        Result dict with keys:
        - 'success': bool - True if result computed, False if any stage failed
        - 'tokens': list | None - Token sequence [12, '+', 3] if produced
        - 'result': int | float | None - Final numeric result if success=True
        - 'error': str | None - Diagnostic error message if success=False
        - 'run_id': str - Unique execution identifier (for logging)
    
    Raises:
        No exceptions (all errors caught and returned in result dict).
    
    Examples:
        result = run('data/sample_expressions/sample_0.png',
                     digit_model, operator_model, device, verbose=True)
        
        if result['success']:
            print(f"Result: {result['result']}")
        else:
            print(f"Error: {result['error']}")
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