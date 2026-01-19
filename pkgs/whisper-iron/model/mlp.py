"""
Feed-forward MLP block for Whisper transformer.
"""

import numpy as np
from typing import Callable

from kernels import linear_bf16, gelu_bf16, add_bf16

BF16_DTYPE = np.float32


class MLP:
    """
    Feed-forward network (MLP) block.

    Structure: Linear -> GELU -> Linear
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize MLP.

        Args:
            d_model: Model dimension (input and output)
            d_ff: Hidden dimension (typically 4 * d_model)
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Weights (will be loaded from checkpoint)
        self.fc1_weight = np.zeros((d_ff, d_model), dtype=BF16_DTYPE)
        self.fc1_bias = np.zeros(d_ff, dtype=BF16_DTYPE)
        self.fc2_weight = np.zeros((d_model, d_ff), dtype=BF16_DTYPE)
        self.fc2_bias = np.zeros(d_model, dtype=BF16_DTYPE)

    def load_weights(
        self,
        fc1_weight: np.ndarray,
        fc1_bias: np.ndarray,
        fc2_weight: np.ndarray,
        fc2_bias: np.ndarray,
    ):
        """Load weights from checkpoint."""
        self.fc1_weight = fc1_weight.astype(BF16_DTYPE)
        self.fc1_bias = fc1_bias.astype(BF16_DTYPE)
        self.fc2_weight = fc2_weight.astype(BF16_DTYPE)
        self.fc2_bias = fc2_bias.astype(BF16_DTYPE)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply MLP.

        Args:
            x: Input of shape (..., d_model)

        Returns:
            Output of shape (..., d_model)
        """
        # First linear: d_model -> d_ff
        h = linear_bf16(x, self.fc1_weight, self.fc1_bias)

        # GELU activation
        h = gelu_bf16(h)

        # Second linear: d_ff -> d_model
        out = linear_bf16(h, self.fc2_weight, self.fc2_bias)

        return out
