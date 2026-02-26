import torch
import torch.nn as nn


class MNISTBaseline(nn.Module):
    """
    WHAT THIS CLASS IS (BIG PICTURE):

    This is a very simple neural network whose only job is to:
        take images of handwritten digits
        and turn each image into 10 numbers (scores for digits 0–9).

    It does NOT:
        - load data
        - train itself
        - compute loss
        - compute accuracy
        - plot anything

    Think of this class as a PIPE:
        images go in on the left
        numbers come out on the right

    Everything else in the project plugs into this pipe.
    """

    def __init__(self, hidden_size: int = 128, num_classes: int = 10) -> None:
        """
        __init__ runs ONCE when you create the model:

            model = MNISTBaseline()

        This is where we DEFINE the pieces of the network
        (layers), but we do NOT run data through them yet.
        """

        # This is mandatory.
        # It sets up internal PyTorch machinery so parameters
        # (weights & biases) are tracked correctly.
        super().__init__()

        # STEP 1: FLATTEN
        # Images come in shaped like:
        #   (batch_size, channels, height, width)
        # Dense (Linear) layers do NOT understand images,
        # so we flatten each image into a long list of numbers.
        #
        # Example:
        #   (1, 28, 28) -> 784 numbers
        #
        # Batch dimension is preserved.
        self.flatten = nn.Flatten()

        # STEP 2: FIRST DENSE LAYER
        # This layer:
        #   - takes the flattened pixels
        #   - combines them using learned weights
        #   - outputs 'hidden_size' numbers
        self.fc1 = nn.Linear(28 * 28, hidden_size)

        # STEP 3: NON-LINEARITY (ReLU)
        # Linear layers alone can only learn straight-line rules.
        #
        # ReLU introduces a decision gate:
        #   - if a value is negative → set it to 0
        #   - if it is positive → keep it
        #
        # This allows the network to learn complex patterns.
        self.relu = nn.ReLU()

        # STEP 4: FINAL DENSE LAYER
        # This maps the hidden representation to class scores.
        #
        # Output shape will be:
        #   (batch_size, num_classes)
        #
        # These are called LOGITS:
        #   raw scores, not probabilities.
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FORWARD PASS (THIS IS THE PIPE WORKING):

        This function describes HOW data flows through the model.

        Input:
            x: a batch of images shaped (N, C, H, W)

        Output:
            logits shaped (N, 10)
        """

        # Turn images into flat vectors so dense layers can use them
        x = self.flatten(x)

        # Combine pixel information into hidden features
        x = self.fc1(x)

        # Remove negative activations to add non-linearity
        x = self.relu(x)

        # Produce one score per digit class (0–9)
        x = self.fc2(x)

        # Return raw scores (logits)
        return x


# ------------------------------------------------------------
# BELOW THIS LINE IS A SANITY CHECK (NOT TRAINING)
# ------------------------------------------------------------
if __name__ == "__main__": #Don’t run the sanity check at import time
# Create the model (builds the pipe)
    model = MNISTBaseline()

# Create a fake batch of MNIST-like images:
#   32 images
#   1 channel (grayscale)
#   28x28 pixels
    x = torch.randn(32, 1, 28, 28)

# Push the batch through the model
    y = model(x)

# If everything is wired correctly, this should be:
#   one row per image
#   one column per digit class
    print(y.shape)  # Expected: torch.Size([32, 10])
