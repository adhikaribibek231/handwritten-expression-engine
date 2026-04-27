"""Dense MNIST baseline.

this is the plain first model in the repo. it flattens each 28x28 digit,
pushes the pixels through one hidden layer, and predicts one of the 10
classes. the training scripts use it as the simple reference point before
moving to the CNN.
"""

import torch
import torch.nn as nn


# step 1 - define the baseline network
class MNISTBaseline(nn.Module):
    """Simple multilayer perceptron for MNIST digits."""

    def __init__(self, hidden_size: int = 128, num_classes: int = 10) -> None:
        """Build the tiny dense stack: flatten -> hidden layer -> class logits."""

        super().__init__()

        # step 1 - flatten the 28x28 image into one long vector
        self.flatten = nn.Flatten()

        # step 2 - learn a compact hidden representation from raw pixels
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()

        # step 3 - map the hidden features to the 10 digit classes
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # this is the whole shape story: 784 -> hidden_size -> 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch of images through the baseline and return raw logits."""

        # step 1 - flatten
        x = self.flatten(x)

        # step 2 - hidden transform
        x = self.fc1(x)
        x = self.relu(x)

        # step 3 - output logits
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Quick shape sanity check.
    model = MNISTBaseline()
    x = torch.randn(32, 1, 28, 28)
    y = model(x)
    print(y.shape)  # Expected: torch.Size([32, 10])
