from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms

# map folder names → class indices
FOLDER_TO_CLASS = {
    'add': '+',
    'sub': '-',
    'mul': '×',
    'div': '÷',
}
CLASS_TO_IDX = {'+': 0, '-': 1, '×': 2, '÷': 3}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
    ),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


class OperatorDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.transform = transform or EVAL_TRANSFORM
        self.samples = []
        root_dir = Path(root_dir)

        per_class: dict[int, list] = {}
        for folder, class_name in FOLDER_TO_CLASS.items():
            class_dir = root_dir / folder
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing folder: {class_dir}")
            images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            idx = CLASS_TO_IDX[class_name]
            per_class[idx] = [(p, idx) for p in images]

        # balance: cap every class to the smallest count so the model
        # never learns to favour one operator over another
        min_count = min(len(v) for v in per_class.values())
        print(f"Balancing to {min_count} samples per class")
        for idx, samples in per_class.items():
            self.samples.extend(samples[:min_count])

        print(f"Loaded {len(self.samples)} total operator samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(str(path)).convert('L')
        img = self.transform(img)
        return img, label


def get_splits(root_dir: Path, val_split: float = 0.2):
    """
    Split into train/val with separate transforms.
    Train gets augmentation, val stays clean — same pattern as digit model.
    """
    # build once just to get the full balanced sample list
    full = OperatorDataset(root_dir, transform=None)
    total      = len(full)
    val_size   = int(total * val_split)
    train_size = total - val_size

    train_subset, val_subset = random_split(
        full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # replace the shared dataset reference with per-split copies
    # so each split gets its own transform without affecting the other
    train_subset.dataset = OperatorDataset(root_dir, transform=TRAIN_TRANSFORM)
    val_subset.dataset   = OperatorDataset(root_dir, transform=EVAL_TRANSFORM)

    print(f"Train: {train_size}  Val: {val_size}")
    return train_subset, val_subset