import torch
import torch.nn as nn


class MNISTBaseline(nn.Module):
    """Simple MNIST MLP: (N, 1, 28, 28) -> logits (N, num_classes)."""

    def __init__(self, hidden_size: int = 128, num_classes: int = 10) -> None:
        """Define a 2-layer dense network with ReLU."""

        super().__init__()

        # Flatten each 28x28 image to a 784-dim vector.
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images."""
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Quick shape sanity check.
    model = MNISTBaseline()
    x = torch.randn(32, 1, 28, 28)
    y = model(x)
    print(y.shape)  # Expected: torch.Size([32, 10])
