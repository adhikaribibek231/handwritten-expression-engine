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

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from torchvision import datasets
from torchvision.transforms import ToTensor

from models.baseline_dense import MNISTBaseline

#DEFINE PROJECT  ROOT
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT /"data"


#Load MNIST
MNIST_TRAIN = datasets.MNIST(root=DATA_DIR, train=True, download =False, transform=ToTensor())
MNIST_TEST = datasets.MNIST(root=DATA_DIR, train=False, download =False, transform=ToTensor())

"""
Train/validation split using a dedicated PyTorch generator.

Using a local RNG ensures the split is reproducible and does not depend
on the global random state, making it robust to refactoring and other
random operations elsewhere in the code.
See: https://discuss.pytorch.org/t/how-to-split-dataset-into-two-considering-fixed-seed-to-ensure-reproducibility-in-pytorch/136514
"""
gen = torch.Generator()
gen.manual_seed(7)
train_set, valid_set = random_split(MNIST_TRAIN,[50000,10000], generator=gen)
test_set = MNIST_TEST

# DataLoaders
batch_size = 64
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True
)
valid_loader = DataLoader(
    valid_set,
    batch_size=batch_size,
    shuffle=False
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False
)
#get a single batch from the training DataLoader
x0, y0 = next(iter(train_loader))

print("batch shape: ", x0.shape)
print("label shape: ", y0.shape)
print("label min/max: ", y0.min().item(), y0.max().item())
print("label dtype (before):", y0.dtype)


# Select device and move data + model to it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Currently using: ", device)
x0= x0.to(device)
y0=y0.to(device).long() #CrossEntropyLoss requires Long targets

#Instantiate the model
model = MNISTBaseline().to(device)

# Forward pass (no gradients, no training)
logits = model(x0)
print("Logits shape: ", logits.shape)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #loss tells you “wrong”; optimizer is what changes weights to reduce loss.

# Compute loss once to verify everything is wired correctly
loss = criterion(logits, y0)
print("loss:", loss.item()) 

# Sanity check: loss must be finite
assert torch.isfinite(loss), "Loss is not finite!"


epochs = 5

for epoch in range(epochs):

    # ------------------------------------------------------------
    # TRAINING PHASE (updates model weights)
    # ------------------------------------------------------------
    model.train()  # enable training behavior

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        # Move batch to the same device as the model
        x = x.to(device)
        y = y.to(device).long()  # CrossEntropyLoss requires Long targets

        # Clear gradients from the previous step
        optimizer.zero_grad()

        # Forward pass: compute raw class scores (logits)
        logits = model(x)

        # Compute loss for this batch
        loss = criterion(logits, y)

        # Backpropagation + parameter update
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy statistics
        train_loss_sum += loss.item() * x.size(0)
        train_correct += (logits.argmax(dim=1) == y).sum().item()
        train_total += x.size(0)

    # Compute average training metrics for the epoch
    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # ------------------------------------------------------------
    # VALIDATION PHASE (no weight updates)
    # ------------------------------------------------------------
    model.eval()  # disable training-only behavior

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    # Disable gradient computation for validation
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            y = y.to(device).long()

            # Forward pass only
            logits = model(x)
            loss = criterion(logits, y)

            # Accumulate validation statistics
            val_loss_sum += loss.item() * x.size(0)
            val_correct += (logits.argmax(dim=1) == y).sum().item()
            val_total += x.size(0)

    # Compute average validation metrics for the epoch
    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    # ------------------------------------------------------------
    # LOG METRICS
    # ------------------------------------------------------------
    print(
        f"Epoch {epoch + 1} | "
        f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
        f"val loss {val_loss:.4f} acc {val_acc:.4f}"
    )

