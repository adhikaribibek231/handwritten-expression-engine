# Getting Started with Calcinator

Calcinator is a **complete, production-ready handwritten arithmetic expression calculator**. This guide will get you up and running in minutes.

---

## Quick Start (< 5 minutes)

### Option 1: Run the Demo (No Setup)

```bash
pip install torch torchvision opencv-python numpy
python app/main.py
```

This runs the full pipeline on 3 pre-loaded sample expressions and displays results.

### Option 2: Run the API Server

```bash
pip install torch torchvision opencv-python numpy fastapi uvicorn python-multipart
python api/server.py
```

Server starts on `http://localhost:8000`

**Test the API:**
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "accept: application/json" \
  -F "file=@data/sample_expressions/sample_0.png"
```

**Response:**
```json
{
  "success": true,
  "tokens": [6, "+", 7],
  "result": 13,
  "error": null,
  "run_id": "20260430_120000_123456"
}
```

### Option 3: Process a Single Image

```bash
python app/main.py data/sample_expressions/sample_1.png
```

---

## System Overview

```
Input Image (grayscale, handwritten arithmetic)
    ↓
Segmentation → Extract individual symbols
    ↓
Classification → Recognize each symbol (digit or operator)
    ↓
Grouping → Combine digits into multi-digit numbers
    ↓
Parsing & Evaluation → Apply precedence, compute result
    ↓
Output: Numeric result or error message
```

**Key Components:**
- **Phase 6**: Image segmentation (classical CV, no ML)
- **Phases 7-8**: Symbol classification (CNNs for digits/operators)
- **Phase 9**: Expression parsing (deterministic, no ML)
- **Phase 10**: End-to-end integration with FastAPI

---

## Bare Minimum Dependencies

**For inference/API:**
```
torch>=2.11.0
torchvision>=0.26.0
opencv-python>=4.13.0.92
numpy>=2.4.3
fastapi>=0.136.1
uvicorn>=0.46.0
python-multipart>=0.0.27
```

**For development/training:**
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
cd /path/to/calcinator
uv sync
```

---

## What Calcinator Can Do

✅ **Recognizes:** Digits 0-9 and operators + - × ÷  
✅ **Handles:** Multi-digit numbers (e.g., 123 + 45)  
✅ **Respects:** Operator precedence (× and ÷ before + and -)  
✅ **Returns:** Explicit errors (low confidence, invalid syntax, division by zero)  
✅ **Provides:** Confidence scores for each recognized symbol  
✅ **Logs:** All execution details for debugging  

### Examples

| Input | Output |
| --- | --- |
| Handwritten: 6 + 7 | Result: 13 ✅ |
| Handwritten: 40 - 9 | Result: 31 ✅ |
| Handwritten: 2 + 3 × 4 | Result: 14 ✅ (not 20, precedence respected) |
| Invalid syntax: + + 3 | Error: "Invalid token count" |
| Low confidence: ambiguous handwriting | Error: "Low confidence — please redraw clearly" |

---

## What Calcinator Cannot Do (V1)

❌ Parentheses (Phase 11)  
❌ Decimals (Phase 12)  
❌ Negative numbers (Phase 12)  
❌ Letters/variables (Phase 13)  
❌ Multiple equations in one image (future)  

---

## Project Structure

```
calcinator/
├─ app/                    # End-to-end pipeline & CLI
├─ api/                    # FastAPI server for HTTP access
├─ models/                 # Neural network definitions
├─ recognition/            # Digit & operator classification
├─ vision/                 # Image segmentation
├─ parser/                 # Expression parsing & evaluation
├─ preprocessing/          # Image preprocessing
├─ checkpoints/            # Pre-trained model weights (needed for inference)
├─ docs/                   # Complete technical documentation
├─ docs/results/           # Detailed phase reports
└─ README.md              # Main documentation
```

**For production inference:** Only need `app/`, `api/`, `checkpoints/`, `models/`, `recognition/`, `vision/`, `preprocessing/`, `parser/`

**For development:** Also need `scripts/`, `tests/`, `data/`, `notebooks/`, `metrics/`, `artifacts/`

---

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install minimal inference dependencies
RUN pip install torch torchvision opencv-python numpy fastapi uvicorn python-multipart

# Copy only necessary files
COPY checkpoints/ ./checkpoints/
COPY models/ ./models/
COPY recognition/ ./recognition/
COPY vision/ ./vision/
COPY preprocessing/ ./preprocessing/
COPY parser/ ./parser/
COPY app/ ./app/
COPY api/ ./api/

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t calcinator:latest .
docker run -p 8000:8000 calcinator:latest
```

### Cloud Deployment

The API is compatible with:
- **AWS ECS/Lambda** (via FastAPI + uvicorn)
- **Google Cloud Run**
- **Azure Container Instances**
- **Heroku** (via Dockerfile)

---

## Understanding Results

### Success Response

```json
{
  "success": true,
  "tokens": [12, "+", 3],
  "result": 15,
  "error": null,
  "run_id": "20260430_120000_123456"
}
```

**Fields:**
- `tokens`: Recognized symbols as a list [operand, operator, operand, ...]
- `result`: Computed numeric result
- `run_id`: Unique execution ID for logging/debugging

### Error Response

```json
{
  "success": false,
  "tokens": null,
  "result": null,
  "error": "Low confidence — please redraw clearly",
  "run_id": "20260430_120000_654321"
}
```

**Common errors:**
- `"Segmentation failed: [details]"` — Image too small, no symbols found
- `"Low confidence — please redraw clearly"` — Handwriting unclear
- `"Invalid expression: Division by zero"` — Parser caught division by zero
- `"Invalid expression: [details]"` — Syntax error in token sequence

---

## Debugging & Logs

### View Execution Logs

After running the pipeline, logs are saved to `artifacts/runs/<run_id>.json`

```bash
# Pretty-print a log file
python -m json.tool artifacts/runs/20260430_120000_123456.json
```

### Debug Segmentation

```bash
python scripts/debug_segmentation.py data/sample_expressions/sample_0.png
```

Generates debug images in `artifacts/phase6/sample_0/`:
- `original.png` — Input image
- `thresholded.png` — Binary after Otsu thresholding
- `boxed_overlay.png` — Detected bounding boxes (most important)
- `crop_0.png`, `crop_1.png`, etc. — Individual symbol crops

---

## Documentation

For detailed technical information:

- **README.md** — Full system overview
- **docs/phases.md** — Phase-by-phase architecture
- **docs/results/** — Detailed reports for each phase:
  - Phase 0: System framing
  - Phase 1-4: Model training & validation
  - Phase 5-7: Perception pipeline
  - Phase 8-10: Complete integration

---

## Performance

**End-to-end accuracy:** 100% on sample expressions (3/3 correct)

**Component accuracies:**
- Digit recognition: 99.30% test accuracy
- Operator recognition: Specialized 4-class CNN
- Segmentation: Robust to variable handwriting
- Parser: 100% (deterministic)

**Operating point:** 97.70% coverage with 99.87% accepted accuracy at confidence threshold 0.90

---

## Support & Issues

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'torch'`
- **Fix:** `pip install torch torchvision`

**Issue:** `FileNotFoundError: checkpoints/cnn_robust/best.pt`
- **Fix:** Ensure checkpoint files exist (they're in the repo)

**Issue:** API returns low confidence error
- **Fix:** Ensure handwriting is clear and well-spaced. See segmentation debug artifacts.

**Issue:** Segmentation detects wrong number of symbols
- **Fix:** Try `scripts/debug_segmentation.py` to visualize. May need to adjust spacing or symbol size.

### Getting Help

- Check `docs/results/phase_06.md` for segmentation issues
- Check `docs/results/phase_07.md` for classification issues  
- Check `docs/results/phase_09.md` for parsing errors

---

## Next Steps

- **Run the demo:** `python app/main.py`
- **Try the API:** `python api/server.py` then curl the `/evaluate` endpoint
- **Read the docs:** Start with `docs/phases.md` for architecture overview
- **Develop locally:** `uv sync` then `uv run python scripts/train_baseline.py` to retrain

---

**Calcinator v1.0 — Complete and production-ready.**
