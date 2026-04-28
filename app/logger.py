"""Structured logging for the calcinator pipeline."""

from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path

LOG_DIR     = Path('artifacts/runs')
FAILURE_DIR = Path('artifacts/failures')


def make_run_id() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def log_run(
    run_id:    str,
    image_path: str,
    stage:     str,
    data:      dict,
    failed:    bool = False,
) -> None:
    """
    Log one stage of the pipeline to artifacts/runs/<run_id>.json.
    If failed=True, also copies the image to artifacts/failures/.
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