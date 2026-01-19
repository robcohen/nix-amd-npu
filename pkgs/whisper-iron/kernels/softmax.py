"""
Softmax kernel for AMD NPU.

NPU Acceleration:
- Uses row-wise softmax kernel for attention
- Falls back to CPU for small tensors
"""

import numpy as np

# Try to import IRON-based NPU kernels
try:
    from .npu import IRON_AVAILABLE
except ImportError:
    IRON_AVAILABLE = False

BF16_DTYPE = np.float32

# Minimum row size for NPU acceleration
NPU_MIN_ROW_SIZE = 64


def softmax_bf16(x: np.ndarray, axis: int = -1, use_npu: bool = True) -> np.ndarray:
    """
    Softmax function with NPU acceleration.

    softmax(x)_i = exp(x_i) / sum_j(exp(x_j))

    For numerical stability, we compute:
    softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

    Args:
        x: Input tensor
        axis: Axis along which to compute softmax (default: -1)
        use_npu: Whether to attempt NPU acceleration

    Returns:
        Softmax probabilities (same shape as x, sums to 1 along axis)
    """
    x = x.astype(BF16_DTYPE)

    # Normalize axis
    if axis < 0:
        axis = x.ndim + axis

    # Check if NPU acceleration is beneficial
    row_size = x.shape[axis]
    can_use_npu = (
        use_npu and
        IRON_AVAILABLE and
        row_size >= NPU_MIN_ROW_SIZE and
        row_size <= 1504 and  # Max supported size
        x.flags['C_CONTIGUOUS']
    )

    if can_use_npu:
        try:
            return _softmax_bf16_npu(x, axis)
        except Exception:
            pass  # Fall through to CPU

    # CPU fallback: numerically stable softmax
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


def _softmax_bf16_npu(x: np.ndarray, axis: int) -> np.ndarray:
    """NPU implementation of softmax."""
    from .npu.softmax_kernel import softmax_iron
    return softmax_iron(x, axis=axis)


def masked_softmax_bf16(
    x: np.ndarray,
    mask: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    Masked softmax for attention.

    Positions where mask is True are set to -inf before softmax,
    resulting in 0 probability after softmax.

    Args:
        x: Input tensor (typically attention scores)
        mask: Boolean mask, True = masked out (set to 0 probability)
        axis: Axis along which to compute softmax

    Returns:
        Masked softmax probabilities
    """
    x = x.astype(BF16_DTYPE)

    # Apply mask: set masked positions to large negative value
    masked_x = np.where(mask, -1e9, x)

    return softmax_bf16(masked_x, axis=axis)


def causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal attention mask.

    Returns a mask where position i can only attend to positions <= i.
    True = masked (cannot attend), False = can attend.

    Args:
        seq_len: Sequence length

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    # Upper triangular (excluding diagonal) = cannot attend
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
