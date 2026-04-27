"""Convolutional MNIST model.

This version keeps the 2D layout of the digit for longer, so the model can
learn strokes, loops, and local shapes before everything is flattened.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# step 1 - define the CNN used in the later phases
class MNISTCNN(nn.Module):
    """Small CNN used in the later MNIST phases."""

    def __init__(self, hidden_size: int = 128, num_classes: int = 10) -> None:
        """Build two conv blocks followed by a small dense classification head."""

        super().__init__()

        # step 1 - read local stroke patterns with convolutions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 32×28×28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64×14×14 after pool

        # step 2 - shrink the feature maps while keeping the strong signals
        self.pool = nn.MaxPool2d(2, 2)                            # halves H,W
        self.dropout = nn.Dropout(p=0.25)

        # step 3 - turn spatial features into 10 class logits
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size)             # 3136 -> hidden
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Turn a batch of digit images into one logit vector per sample."""

        # step 1 - extract local features and downsample twice
        x = self.pool(F.relu(self.conv1(x)))  # 32×14×14
        x = self.pool(F.relu(self.conv2(x)))  # 64×7×7

        # step 2 - flatten the spatial features into one vector
        x = self.flatten(x)                   # 3136

        # step 3 - classify with a small dense head
        x = F.relu(self.fc1(x))               # hidden_size
        x = self.dropout(x)
        x = self.fc2(x)                       # logits (10)

        return x


if __name__ == "__main__":
    model = MNISTCNN()
    x = torch.randn(32, 1, 28, 28)
    y = model(x)
    print(y.shape)
