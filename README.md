# Calcinator
## Handwritten Mathematical Expression Intelligence System

Handwritten arithmetic expression calculator. Calcinator takes an image of a
simple handwritten expression, recognizes the symbols, groups digits into
numbers, and evaluates the result with deterministic parser logic.

The project keeps machine-learning perception separate from arithmetic
evaluation: CNNs recognize symbols with confidence scores, then rule-based code
validates and evaluates the token sequence.

## What It Supports

- Digits `0-9`
- Operators `+`, `-`, `×`, `÷`
- Multi-digit integers such as `12 + 3`
- Standard operator precedence: multiplication and division before addition and
  subtraction
- Explicit errors for low confidence, invalid syntax, and division by zero
- CLI demo, single-image CLI mode, and FastAPI HTTP endpoint

## Current Results

| Area | Result |
| --- | --- |
| Digit CNN test accuracy | 99.30% |
| Robust digit benchmark threshold | 0.90 operating point |
| Accepted digit accuracy at benchmark threshold | 99.87% |
| Accepted digit coverage at benchmark threshold | 97.70% |
| Runtime confidence thresholds | digits 0.75, operators 0.65 |
| Operator CNN validation accuracy | 100.00% final validation epoch |
| End-to-end demo samples | 3/3 correct |

The end-to-end demo currently validates these sample expressions:

| Image | Expression | Expected result |
| --- | --- | --- |
| `data/sample_expressions/sample_0.png` | `6 + 7` | `13` |
| `data/sample_expressions/sample_2.png` | `40 - 9` | `31` |
| `data/sample_expressions/sample_3.png` | `7 - 1` | `6` |

## Requirements

- Python 3.13
- `uv` for dependency management

Core dependencies are declared in `pyproject.toml`:

- PyTorch / torchvision
- OpenCV
- NumPy
- FastAPI
- Uvicorn
- python-multipart

The trained checkpoints needed for inference are included under
`checkpoints/`:

- `checkpoints/cnn_robust/best.pt`
- `checkpoints/operator_cnn/best.pt`

## Clone And Install

```bash
git clone https://github.com/adhikaribibek231/calcinator.git
cd calcinator
uv sync
```

## Run The Demo

```bash
uv run python app/main.py
```

This runs the bundled sample expressions through the full pipeline and prints
the predicted tokens and result.

## Run A Single Image

```bash
uv run python app/main.py data/sample_expressions/sample_0.png
```

## Run The API

```bash
uv run uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Evaluate an image:

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "accept: application/json" \
  -F "file=@data/sample_expressions/sample_0.png"
```

Example response:

```json
{
  "success": true,
  "tokens": ["6", "+", "7"],
  "result": 13,
  "error": null
}
```

## Run Tests

```bash
uv run pytest
```

## Train Models

Training is optional for normal use because the inference checkpoints are
already included.

```bash
uv run python scripts/train_cnn_robust.py
uv run python scripts/train_operator_cnn.py
```

Useful evaluation scripts:

```bash
uv run python scripts/analyze_failures.py
uv run python scripts/evaluate_thresholds.py
```

## Project Structure

```text
api/             FastAPI server
app/             End-to-end pipeline and CLI entry point
checkpoints/     Trained model weights used for inference
data/            Sample images and local datasets
metrics/         Training and evaluation CSVs
models/          Neural network definitions
parser/          Expression validation and arithmetic evaluation
preprocessing/   Image preprocessing utilities
recognition/     Digit and operator recognition
scripts/         Training, debugging, and evaluation scripts
tests/           Test suite
vision/          Segmentation and classical CV logic
```

## Limitations

V1 intentionally does not support:

- Parentheses
- Decimals
- Negative numbers
- Variables or letters
- Multiple expressions in one image

Segmentation also works best when symbols are clearly separated. Touching
symbols, unusual handwriting, or broken symbols can trigger low-confidence
rejection instead of a guessed result.

## License

No license file is currently defined.
