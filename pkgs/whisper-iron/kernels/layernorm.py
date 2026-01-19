"""
Layer normalization kernel for AMD NPU.

NPU Acceleration:
- Uses vectorized reduction for mean/variance
- Falls back to CPU for unsupported dimensions
"""

import numpy as np
from typing import Optional

# Try to import IRON-based NPU kernels
try:
    from .npu import IRON_AVAILABLE
except ImportError:
    IRON_AVAILABLE = False

BF16_DTYPE = np.float32

# Supported normalization dimensions
NPU_SUPPORTED_DIMS = [64, 128, 256, 384, 512, 768, 1024, 1536]


def layernorm_bf16(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    eps: float = 1e-5,
    use_npu: bool = True,
) -> np.ndarray:
    """
    Layer normalization with NPU acceleration.

    LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias

    Normalizes over the last dimension.

    Args:
        x: Input tensor of shape (..., normalized_shape)
        weight: Scale parameter (gamma) of shape (normalized_shape,)
        bias: Shift parameter (beta) of shape (normalized_shape,), optional
        eps: Small constant for numerical stability
        use_npu: Whether to attempt NPU acceleration

    Returns:
        Normalized tensor of same shape as x
    """
    x = x.astype(BF16_DTYPE)
    weight = weight.astype(BF16_DTYPE)
    if bias is not None:
        bias = bias.astype(BF16_DTYPE)

    # Check if NPU acceleration is beneficial
    norm_dim = x.shape[-1]
    can_use_npu = (
        use_npu and
        IRON_AVAILABLE and
        norm_dim in NPU_SUPPORTED_DIMS and
        x.flags['C_CONTIGUOUS']
    )

    if can_use_npu:
        try:
            return _layernorm_bf16_npu(x, weight, bias, eps)
        except Exception:
            pass  # Fall through to CPU

    # CPU fallback
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = x_norm * weight
    if bias is not None:
        out = out + bias

    return out


def _layernorm_bf16_npu(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    eps: float,
) -> np.ndarray:
    """NPU implementation of layer normalization."""
    from .npu.layernorm_kernel import layernorm_iron
    if bias is None:
        bias = np.zeros_like(weight)
    return layernorm_iron(x, weight, bias, axis=-1, eps=eps)


class LayerNormBF16:
    """
    Layer normalization module.

    Stores weight and bias parameters.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Initialize layer norm.

        Args:
            normalized_shape: Size of the last dimension to normalize
            eps: Numerical stability constant
        """
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Initialize parameters (will be loaded from checkpoint)
        self.weight = np.ones(normalized_shape, dtype=BF16_DTYPE)
        self.bias = np.zeros(normalized_shape, dtype=BF16_DTYPE)

    def load_weights(self, weight: np.ndarray, bias: np.ndarray):
        """Load weight and bias from checkpoint."""
        self.weight = weight.astype(BF16_DTYPE)
        self.bias = bias.astype(BF16_DTYPE)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        return layernorm_bf16(x, self.weight, self.bias, self.eps)
