"""Calcinator — end-to-end handwritten expression evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from app.pipeline import load_models, run

device                         = torch.device('cpu')
digit_model, operator_model    = load_models(device)

# ── Demo run ──────────────────────────────────────────────────────────────────
DEMO_SAMPLES = [
    ('data/sample_expressions/sample_0.png', '6+7',  13),
    ('data/sample_expressions/sample_2.png', '40-9', 31),
    ('data/sample_expressions/sample_3.png', '7-1',  6),
]

print("\n── Calcinator demo ──────────────────────────────────\n")

passed = 0
failed = 0

for image_path, expression, expected in DEMO_SAMPLES:
    print(f"Expression: {expression}")
    output = run(image_path, digit_model, operator_model, device, verbose=True)

    if output['success']:
        correct = output['result'] == expected
        status  = '✅' if correct else '❌ wrong answer'
        print(f"  result: {output['result']}  {status}")
        if correct:
            passed += 1
        else:
            failed += 1
    else:
        print(f"  ✗ {output['error']}")
        failed += 1

    print()

print(f"── Results: {passed}/{passed+failed} passed ──────────────────────────\n")


# ── Single image mode (CLI arg) ───────────────────────────────────────────────
if __name__ == '__main__' and len(sys.argv) > 1:
    image_path = sys.argv[1]
    print(f"\nRunning on: {image_path}")
    output = run(image_path, digit_model, operator_model, device, verbose=True)
    if output['success']:
        print(f"\nResult: {output['result']}")
    else:
        print(f"\nFailed: {output['error']}")