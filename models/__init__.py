"""Model definitions for digit and operator classification."""

from models.baseline_dense import MNISTBaseline
from models.cnn_mnist import MNISTCNN

__all__ = ["MNISTBaseline", "MNISTCNN"]
