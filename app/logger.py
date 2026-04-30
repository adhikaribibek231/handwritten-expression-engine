"""Structured logging for the Calcinator end-to-end pipeline (Phase 10).

This module provides infrastructure for recording all stages of the expression
recognition and evaluation process to disk for debugging, monitoring, and
post-mortem analysis. Every run produces a unique run ID and a JSON log file
containing stage-by-stage progress and any failure information.

The logging system serves three purposes:
1. Debugging: Inspect exactly what happened at each stage for a given image
2. Monitoring: Analyze aggregate statistics across many runs
3. Failure case collection: Automatically save images that caused errors

For detailed implementation, see docs/results/phase_10.md.
"""

from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path

LOG_DIR     = Path('artifacts/runs')
FAILURE_DIR = Path('artifacts/failures')


def make_run_id() -> str:
    """Generate a unique run identifier using timestamp and microseconds.
    
    Format: YYYYMMDD_HHMMSS_ffffff (date_time_microseconds)
    This ensures uniqueness even when multiple runs happen in rapid succession.
    
    Returns:
        str: Unique run ID like '20260429_123456_789012'
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def log_run(
    run_id:    str,
    image_path: str,
    stage:     str,
    data:      dict,
    failed:    bool = False,
) -> None:
    """Record one pipeline stage to a JSON log file.
    
    Called by app/pipeline.py after each stage completes. Multiple calls with
    the same run_id append to the same log file, building a complete execution
    trace.
    
    Log files are stored at:
        artifacts/runs/<run_id>.json
    
    Each log file contains:
        {
            'run_id': str,
            'image': str (path),
            'stages': [
                {
                    'stage': str (e.g., 'segmentation', 'classification'),
                    'data': dict (stage-specific results),
                    'failed': bool
                },
                ...
            ]
        }
    
    If a stage fails (failed=True), the input image is also copied to:
        artifacts/failures/<run_id>_<image_name>
    
    This enables easy inspection of problematic images without re-running.
    
    Args:
        run_id: Unique execution identifier from make_run_id()
        image_path: Path to input image being processed
        stage: Pipeline stage name ('segmentation', 'classification', etc.)
        data: Stage-specific results (dict, will be JSON-serialized)
        failed: Whether this stage encountered an error
    
    Returns:
        None (writes to disk)
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f'{run_id}.json'

    # load existing log or start fresh
    if log_file.exists():
        with log_file.open() as f:
            log = json.load(f)
    else:
        log = {'run_id': run_id, 'image': image_path, 'stages': []}

    log['stages'].append({
        'stage':  stage,
        'data':   data,
        'failed': failed,
    })

    with log_file.open('w') as f:
        json.dump(log, f, indent=2, default=str)

    # save failure cases for later inspection
    if failed:
        FAILURE_DIR.mkdir(parents=True, exist_ok=True)
        src = Path(image_path)
        if src.exists():
            dst = FAILURE_DIR / f'{run_id}_{src.name}'
            shutil.copy(src, dst)