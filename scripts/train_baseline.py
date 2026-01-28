"""
=======================
TRAINING CONTRACT (Phase 2)
=======================

Status
------
As of the current repo state, this script contains only this contract.
Training/evaluation code and training outputs are not implemented yet.
This contract is grounded in Phase 1 EDA (see `notebooks/01_mnist_exploration.ipynb`).

Dataset
-------
source: torchvision.datasets.MNIST (download=True)
root: <PROJECT_ROOT>/data/
train split source: torchvision MNIST train=True (60,000 samples)
test split source:  torchvision MNIST train=False (10,000 samples)

Splits
------
train: 50,000  (subset of torchvision train split)
val:   10,000  (subset of torchvision train split)
test:  10,000  (official torchvision test split)
split method: torch.utils.data.random_split
split seed: 42 (torch.Generator().manual_seed)
stratification: none (Phase 1 verified class balance is sufficiently uniform)

Preprocessing
-------------
transform: torchvision.transforms.ToTensor()
value range: [0, 1]
normalization: none (no mean/std normalization applied)
reshape rule:
  - per sample: 1 x 28 x 28 (grayscale channel preserved)
  - per batch:  N x 1 x 28 x 28 (DataLoader output)
flattening: none in DataLoader (handled explicitly in model if needed)

Sanity Verification (Phase 1)
-----------------------------
Verified via direct tensor inspection in the notebook:
  - type: torch.Tensor
  - dtype: torch.float32
  - shape: (1, 28, 28)
  - min/max: ~0.0 / ~1.0
  - device: cpu
  - contiguous memory layout

DataLoader
----------
batch_size: 64 (used in Phase 1 notebook for inspection)
shuffle:
  - train: True
  - val/test: False

Evaluation
----------
task: 10-class digit classification (0–9)
prediction: argmax over model logits
accuracy (top-1): mean(pred == target)
validation cadence: once per epoch (planned)
test evaluation: run after training on best checkpoint (planned)

Outputs (relative to repo root)
-------------------------------
Phase 1 artifacts (present):
  - artifacts/phase1/random_samples.png
  - artifacts/phase1/one_per_digit.png
  - artifacts/phase1/mean_per_class.png
  - artifacts/phase1/pixel_histogram.png
  - artifacts/phase1/debug/*.png

Phase 2 outputs (planned, not yet implemented):
  -metrics/baseline_dense.csv

  -metrics/baseline_dense_curves.png

  -checkpoints/baseline_dense_best.pt #best.pt = checkpoint with highest validation accuracy (tie-breaker: lowest validation loss).

  -checkpoints/baseline_dense_last.pt(optional)

Reproducibility
---------------
random seed: 42 (data split + torch randomness when set globally)
note: GPU determinism settings must be made explicit if GPU is used.
"""
