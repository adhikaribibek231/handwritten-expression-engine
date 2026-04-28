"""FastAPI server wrapping the calcinator pipeline."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import load_models, run

app = FastAPI(title="Calcinator API")

# allow requests from the React Native app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load models once at startup — not on every request
device                      = torch.device("cpu")
digit_model, operator_model = load_models(device)

print("Models loaded. Server ready.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    """
    Receive an image, run the pipeline, return the result.

    Returns:
        {
            "success": true,
            "tokens": [6, "+", 7],
            "result": 13,
            "error": null
        }
    """
    # save upload to a temp file — pipeline expects a file path
    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    output = run(tmp_path, digit_model, operator_model, device, verbose=False)

    # clean up temp file
    Path(tmp_path).unlink(missing_ok=True)

    return {
        "success": output["success"],
        "tokens":  [str(t) for t in output["tokens"]] if output["tokens"] else None,
        "result":  output["result"],
        "error":   output["error"],
    }