# Phase 10 — End-to-End Integration

## Objective

Phases 1–9 have each delivered a focused subsystem: data inspection, model training, preprocessing, segmentation, classification, grouping, and expression evaluation. Phase 10 integrates all of these into a single, continuous end-to-end system that accepts a raw handwritten expression image and produces either a correct numeric result or a clear, actionable error message.

The goal is to verify that the system works as a cohesive whole, with proper instrumentation and logging at each stage to enable debugging, validation, and production monitoring.

## Architecture Overview

The complete pipeline is implemented across two main modules:

### `app/pipeline.py` — Core Pipeline

This is the heart of the system. It orchestrates all phases in strict sequence:

```
Input Image
    ↓ (Phase 6)
Segmentation
    ├─ Load & threshold expression image
    ├─ Find symbol contours
    ├─ Filter and sort bounding boxes
    └─ Extract individual crops → [crop₀, crop₁, ...]
    ↓ (Phase 7/8)
Classification
    ├─ Load digit CNN and operator CNN
    ├─ Route each crop to correct classifier
    └─ Produce (value, confidence) pairs
    ↓
Confidence Validation
    ├─ Check all values meet threshold
    └─ Reject if any too low → error return
    ↓ (Phase 8 continued)
Tokenization
    ├─ Group consecutive digits into multi-digit numbers
    └─ Produce token sequence [12, '+', 3]
    ↓ (Phase 9)
Parsing & Evaluation
    ├─ Validate token syntax
    ├─ Apply operator precedence
    └─ Compute numeric result
    ↓
Result
    ├─ success: True/False
    ├─ result: int | float | None
    ├─ error: str | None
    └─ run_id: str (for logging)
```

### `app/main.py` — Entry Point and CLI

Provides a simple interface for running the system:

**Demo mode:** Runs the pipeline on pre-selected sample images to verify basic functionality.

**Single-image mode:** Accepts an image path as a command-line argument and processes it.

## Implementation Details

### Module: `app/pipeline.py`

#### Function: `load_models(device: torch.device) -> tuple`

Initializes and returns both the digit CNN and operator CNN models. Both models are loaded in evaluation mode with no gradient tracking. The device parameter allows execution on CPU or GPU.

```python
digit_model, operator_model = load_models(device)
```

#### Function: `run(image_path, digit_model, operator_model, device, verbose=True) -> dict`

This is the core orchestration function that runs the full pipeline on a single image. It returns a structured result dictionary:

```python
{
    'success': bool,           # True if result computed, False if any stage failed
    'tokens': list | None,     # Token sequence [12, '+', 3] if produced
    'result': int | float | None,  # Final numeric result if success=True
    'error': str | None,       # Diagnostic error message if success=False
    'run_id': str,            # Unique identifier for this run (for logging)
}
```

**Pipeline stages:**

**Stage 1: Segmentation**
- Calls `segment_expression(image_path)` from Phase 6
- Returns: (boxes, crops, binary)
- On failure: Returns error dict with reason (file not found, no symbols detected, etc.)
- Logs: Number of symbols detected, bounding box coordinates

**Stage 2: Classification**
- Calls `classify_all(crops, digit_model, operator_model, device)` from Phase 8
- Returns: List of `(value, confidence)` tuples
- Each value is either an int (0-9 for digits, or the operator string '+', '-', '×', '÷')
- Logs: Raw classification results as `(value_str, confidence)` pairs

**Stage 3: Confidence Validation**
- Calls `is_low_confidence(results)` from Phase 8
- Checks all confidence values against thresholds (0.75 for digits, 0.65 for operators)
- On failure: Returns error "Low confidence — please redraw clearly"
- On success: Proceeds to tokenization
- Logs: Confidence check result

**Stage 4: Tokenization**
- Calls `build_token_sequence(results)` from Phase 8
- Groups consecutive digit predictions into multi-digit numbers
- Produces token sequence [12, '+', 3] ready for parsing
- Logs: Final token sequence

**Stage 5: Parsing & Evaluation**
- Calls `parse_and_evaluate(tokens)` from Phase 9
- Validates token syntax and applies operator precedence
- Computes final numeric result
- On success: Returns complete result dict with `success=True`, `result=value`
- On failure (e.g., invalid expression): Returns result dict with `success=False`, `error=message`
- Logs: Final result or error reason

#### Logging System

The logging function `log(stage, data, failed=False)` provides:

- **Console output** (if `verbose=True`): Prints stage progress and results
- **Persistent logging** (via `log_run()`): Stores structured logs to disk under `artifacts/runs/`

Each log entry captures:
- Timestamp and run ID
- Image path
- Processing stage
- Stage-specific data (e.g., segmentation: number of symbols; evaluation: final result)
- Whether the stage failed
- Traceback details if an exception occurred

### Module: `app/main.py`

#### Demo Mode

When run without arguments, executes a hardcoded suite of test images:

```python
DEMO_SAMPLES = [
    ('data/sample_expressions/sample_0.png', '6+7',  13),
    ('data/sample_expressions/sample_2.png', '40-9', 31),
    ('data/sample_expressions/sample_3.png', '7-1',  6),
]
```

For each sample:
- Runs the full pipeline
- Compares result to expected value
- Displays ✅ if correct, ❌ if wrong answer, ✗ if error
- Accumulates pass/fail counts

**Usage:**
```bash
python app/main.py
```

**Expected output:**
```
── Calcinator demo ──────────────────────────────

Expression: 6+7
  [segmentation] {'num_symbols': 3, 'boxes': [...]}
  [classification] {'raw': [('6', 0.98), ('+', 0.92), ('7', 0.97)]}
  [tokenization] {'tokens': [6, '+', 7]}
  [evaluation] {'result': 13}
  result: 13  ✅

── Results: 3/3 passed ──────────────────────────
```

#### Single-Image Mode

Accepts an image file path as a command-line argument:

```bash
python app/main.py data/sample_expressions/sample_0.png
```

Processes the image and displays the result or error.

## Supporting Modules

### `app/logger.py`

Provides logging infrastructure:

**`make_run_id() -> str`**
Generates a unique run identifier using timestamp and random ID for deduplication.

**`log_run(run_id, image_path, stage, data, failed=False)`**
Persists logs to disk under `artifacts/runs/<run_id>.json` with structured data.

### `app/__init__.py`

Exports public API: Currently minimal, may be extended for library usage in future versions.

## Data Flow Example

**Input:** Image file `data/sample_expressions/sample_0.png` containing handwritten `6 + 7`

**Execution trace:**

```
run('data/sample_expressions/sample_0.png', digit_model, operator_model, device, verbose=True)

[segmentation] {'num_symbols': 3, 'boxes': [(10, 20, 15, 25), (35, 18, 18, 28), (60, 21, 14, 24)]}
  → Successfully detected 3 symbols

[classification] {'raw': [('6', 0.981), ('+', 0.923), ('7', 0.967)]}
  → Digit model: 0.981 confidence on first crop (6)
  → Operator model: 0.923 confidence on second crop (+)
  → Digit model: 0.967 confidence on third crop (7)

[tokenization] {'tokens': [6, '+', 7]}
  → No digit grouping needed (single-digit numbers)

[evaluation] {'result': 13}
  → Token validation: ✓ odd length, alternating operands/operators
  → Precedence evaluation: 6 + 7 = 13
  → Final result: 13

Return:
{
    'success': True,
    'tokens': [6, '+', 7],
    'result': 13,
    'error': None,
    'run_id': '20260429_123456_789012'
}
```

## Error Handling and Diagnostics

The system handles errors at each stage and provides clear, actionable feedback:

**Segmentation errors:**
- File not found → "Segmentation failed: [FileNotFoundError details]"
- No symbols detected → "No symbols found in image"

**Classification errors:**
- Low confidence → "Low confidence — please redraw clearly"

**Parsing/Evaluation errors:**
- Invalid token syntax → "Invalid expression: [ExpressionError details]"
- Division by zero → "Invalid expression: Division by zero"

**Logging:** All failures are logged with run ID for post-mortem analysis.

## Design Decisions

1. **Sequential, not parallel:** Stages execute strictly in order. This ensures clear error attribution and deterministic behavior.

2. **Structured result dictionary:** The return value is always a dict with consistent keys, making programmatic handling (success/failure, result extraction) straightforward.

3. **Dual logging:** Console output (for human operators) and persistent disk logs (for analytics/debugging) serve different needs.

4. **Stage-specific data in logs:** Each stage logs only relevant information (segmentation logs box count, classification logs confidence values), making logs concise and actionable.

5. **Unique run IDs:** Every execution is tagged with a unique run ID, enabling correlation across multiple logging systems and supporting audit trails.

6. **Verbose flag:** Demo and testing use `verbose=True` for immediate feedback. Production deployments could set `verbose=False` and read only the result dict.

## Validation and Testing

### Demo Suite

The demo mode (run without arguments) validates the entire pipeline on three representative test images. This serves as:
- Smoke test for deployment validation
- Performance baseline
- Quick regression check

### Comprehensive Pipeline Test

Running `app/main.py` should show 3/3 passed, indicating all stages work correctly on real handwritten input.

## Artifacts Generated

- `app/main.py` — Entry point and CLI interface
- `app/pipeline.py` — Core orchestration logic
- `app/logger.py` — Logging infrastructure
- `app/__init__.py` — Public API exports
- `artifacts/runs/<run_id>.json` — Persistent logs per execution (created at runtime)

## Integration Status

Phase 10 represents the completion of the core v1 system:

| Phase | Subsystem | Status |
| --- | --- | --- |
| 0 | System specification | ✅ Complete |
| 1 | Data inspection | ✅ Complete |
| 2–4 | Perception: baseline, CNN, robustness | ✅ Complete |
| 5 | Perception: preprocessing | ✅ Complete |
| 6 | Vision: segmentation | ✅ Complete |
| 7–8 | Recognition: digit & operator classification | ✅ Complete |
| 9 | Reasoning: parsing & evaluation | ✅ Complete |
| 10 | Integration: end-to-end pipeline | ✅ Complete |

The system is now ready for:
- Deployment on real handwritten input
- Performance monitoring via run logs
- User feedback collection for future improvements
- Extensions (Phase 11+): parentheses, decimals, negative numbers, etc.

## Conclusion

Phase 10 completes the v1 system by integrating all previous phases into a single, well-logged, instrumented end-to-end pipeline. The system accepts raw handwritten expression images, processes them through a chain of perception and reasoning stages, and produces correct numeric results or clear diagnostic errors.

The architecture maintains the hard boundary between perception (ML, uncertain) and reasoning (rule-based, deterministic), ensuring both that ML errors are caught before they propagate to deterministic computation, and that symbolic processing is never contaminated by probabilistic approximation.

All intermediate results are logged for debugging and analytics. The demo suite provides quick regression validation. The codebase is now ready for production deployment, user testing, and iterative improvement.

## Next Phases (Extensions)

Phase 11 onwards should focus on extending v1 to support additional features:
- **Parentheses:** Require recursive precedence handling
- **Decimals:** Extend number parsing and tokenization
- **Negative numbers:** Add unary negation operator
- **EMNIST:** Support uppercase and lowercase letters for algebraic expressions
- **Sequence models:** Replace multi-digit detection with CNN+RNN for handwriting recognition
- **User feedback:** Implement correction loop for misclassifications
