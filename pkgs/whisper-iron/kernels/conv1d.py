"""
1D Convolution kernel for AMD NPU.

Used in Whisper audio encoder frontend.

NPU Acceleration:
- Specialized kernels for Whisper layer configurations
- Layer 1: (80, 384, k=3, s=1, p=1)
- Layer 2: (384, 384, k=3, s=2, p=1)
"""

import numpy as np
from typing import Optional, Tuple

# Try to import IRON-based NPU kernels
try:
    from .npu import IRON_AVAILABLE
except ImportError:
    IRON_AVAILABLE = False

BF16_DTYPE = np.float32


def conv1d_bf16(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: int = 1,
    padding: int = 0,
    use_npu: bool = True,
) -> np.ndarray:
    """
    1D convolution with NPU acceleration.

    Args:
        x: Input of shape (batch, in_channels, length)
        weight: Kernel of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias of shape (out_channels,)
        stride: Convolution stride
        padding: Zero-padding on both sides
        use_npu: Whether to attempt NPU acceleration

    Returns:
        Output of shape (batch, out_channels, out_length)
        where out_length = (length + 2*padding - kernel_size) // stride + 1
    """
    x = x.astype(BF16_DTYPE)
    weight = weight.astype(BF16_DTYPE)

    in_channels = x.shape[1] if x.ndim == 3 else x.shape[0]
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]

    # Check if NPU acceleration is feasible
    # Support any kernel_size=3 conv with reasonable dimensions
    can_use_npu = (
        use_npu and
        IRON_AVAILABLE and
        kernel_size == 3 and  # Only kernel_size=3 supported currently
        x.flags['C_CONTIGUOUS']
    )

    if can_use_npu:
        try:
            return _conv1d_bf16_npu(x, weight, bias, stride, padding)
        except Exception:
            pass  # Fall through to CPU

    # CPU fallback (im2col implementation)

    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape

    # Apply padding
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')
        length = x.shape[2]

    # Calculate output length
    out_length = (length - kernel_size) // stride + 1

    # Naive convolution (can be optimized with im2col + matmul)
    out = np.zeros((batch_size, out_channels, out_length), dtype=BF16_DTYPE)

    for b in range(batch_size):
        for oc in range(out_channels):
            for i in range(out_length):
                start = i * stride
                end = start + kernel_size
                # Sum over input channels and kernel positions
                out[b, oc, i] = np.sum(x[b, :, start:end] * weight[oc, :, :])

    if bias is not None:
        out = out + bias.astype(BF16_DTYPE).reshape(1, -1, 1)

    return out


def _conv1d_bf16_npu(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    stride: int,
    padding: int,
) -> np.ndarray:
    """NPU implementation of 1D convolution."""
    from .npu.conv1d_kernel import conv1d_iron
    return conv1d_iron(x, weight, bias if bias is not None else np.zeros(weight.shape[0], dtype=x.dtype),
                       stride=stride, padding=padding)


def conv1d_im2col_bf16(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """
    1D convolution using im2col + matmul.

    This is more efficient and maps better to NPU matmul kernels.

    Args:
        x: Input of shape (batch, in_channels, length)
        weight: Kernel of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias of shape (out_channels,)
        stride: Convolution stride
        padding: Zero-padding on both sides

    Returns:
        Output of shape (batch, out_channels, out_length)
    """
    x = x.astype(BF16_DTYPE)
    weight = weight.astype(BF16_DTYPE)

    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape

    # Apply padding
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')
        length = x.shape[2]

    # Calculate output length
    out_length = (length - kernel_size) // stride + 1

    # im2col: Extract sliding windows and reshape to matrix
    # Shape: (batch, in_channels * kernel_size, out_length)
    col = np.zeros((batch_size, in_channels * kernel_size, out_length), dtype=BF16_DTYPE)

    for i in range(out_length):
        start = i * stride
        end = start + kernel_size
        col[:, :, i] = x[:, :, start:end].reshape(batch_size, -1)

    # Reshape weight: (out_channels, in_channels * kernel_size)
    weight_reshaped = weight.reshape(out_channels, -1)

    # Matmul: (out_channels, in_channels*kernel_size) @ (batch, in_channels*kernel_size, out_length)
    # -> (batch, out_channels, out_length)
    # Using einsum: 'ok,bkl->bol' where o=out_channels, k=in*kernel, l=out_length
    out = np.einsum('ok,bkl->bol', weight_reshaped, col)

    if bias is not None:
        out = out + bias.astype(BF16_DTYPE).reshape(1, -1, 1)

    return out


class Conv1dBF16:
    """
    1D Convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights (will be loaded from checkpoint)
        self.weight = np.zeros((out_channels, in_channels, kernel_size), dtype=BF16_DTYPE)
        self.bias = np.zeros(out_channels, dtype=BF16_DTYPE)

    def load_weights(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        """Load weights from checkpoint."""
        self.weight = weight.astype(BF16_DTYPE)
        if bias is not None:
            self.bias = bias.astype(BF16_DTYPE)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply convolution."""
        return conv1d_im2col_bf16(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding
        )
