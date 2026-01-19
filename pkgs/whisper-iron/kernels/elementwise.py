"""
Element-wise operations for AMD NPU.

Includes: add, gelu, and other activation functions.
Uses MLIR-AIE/IRON when available, falls back to numpy.

NPU Acceleration Strategy:
- Small tensors (<1024 elements): CPU (transfer overhead dominates)
- Large tensors (>=1024 elements): NPU when available
- Broadcasts: CPU (complex memory patterns)
"""

import numpy as np
from typing import Dict, Tuple, Optional

# Try to import NPU runtime (legacy MLIR dialect approach)
try:
    from .npu_runtime import (
        get_runtime, npu_available, generate_elementwise_mlir,
        MLIR_AIE_AVAILABLE
    )
except ImportError:
    MLIR_AIE_AVAILABLE = False

    def npu_available():
        return False

# Try to import IRON-based NPU kernels (modern approach)
try:
    from .npu import IRON_AVAILABLE
except ImportError:
    IRON_AVAILABLE = False

BF16_DTYPE = np.float32

# Minimum size for NPU acceleration (below this, CPU is faster)
NPU_MIN_SIZE = 1024


class ElementwiseKernel:
    """
    Base class for elementwise NPU kernels.

    Compiles and caches kernels for different sizes.
    """

    # Shared cache across all elementwise ops
    _kernel_cache: Dict[Tuple[str, int], str] = {}

    def __init__(self, op: str, size: int, use_npu: bool = True):
        self.op = op
        self.size = size
        self.use_npu = use_npu and npu_available()
        self.tile_size = 1024

        # Round up to tile boundary
        self.size_padded = ((size + self.tile_size - 1) // self.tile_size) * self.tile_size

        if self.use_npu:
            self._init_npu()

    def _init_npu(self):
        """Initialize NPU kernel."""
        key = (self.op, self.size_padded)

        if key not in ElementwiseKernel._kernel_cache:
            runtime = get_runtime()

            mlir_source = generate_elementwise_mlir(
                self.size_padded,
                self.op,
                dtype="bf16",
            )

            kernel_name = f"{self.op}_{self.size_padded}"
            xclbin_path, insts_path = runtime.compile_kernel(kernel_name, mlir_source)

            runtime.load_kernel(
                kernel_name,
                xclbin_path,
                insts_path,
                input_shapes=[(self.size_padded,)],
                input_dtypes=[np.float16],
                output_shape=(self.size_padded,),
                output_dtype=np.float16,
            )

            ElementwiseKernel._kernel_cache[key] = kernel_name

        self.kernel_name = ElementwiseKernel._kernel_cache[key]


def add_bf16(a: np.ndarray, b: np.ndarray, use_npu: bool = True) -> np.ndarray:
    """
    Element-wise addition: c = a + b

    Supports broadcasting. Uses NPU for large contiguous arrays.

    Args:
        a: First operand
        b: Second operand
        use_npu: Whether to attempt NPU acceleration

    Returns:
        c: Element-wise sum
    """
    a = a.astype(BF16_DTYPE)
    b = b.astype(BF16_DTYPE)

    # Check if NPU acceleration is beneficial
    # Requirements: same shape, large enough, contiguous
    can_use_npu = (
        use_npu and
        (IRON_AVAILABLE or npu_available()) and
        a.shape == b.shape and
        a.size >= NPU_MIN_SIZE and
        a.flags['C_CONTIGUOUS'] and
        b.flags['C_CONTIGUOUS']
    )

    if can_use_npu:
        try:
            return _add_bf16_npu(a, b)
        except Exception:
            pass  # Fall through to CPU

    # CPU fallback (handles broadcasting automatically)
    return (a + b).astype(BF16_DTYPE)


def _add_bf16_npu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NPU implementation of vector addition."""
    if IRON_AVAILABLE:
        # Use modern IRON-based kernel
        from .npu.vector_add import vector_add_iron
        return vector_add_iron(a, b)
    elif npu_available():
        # Use legacy MLIR dialect approach
        orig_shape = a.shape
        a_flat = a.ravel()
        b_flat = b.ravel()
        size = a_flat.size

        kernel = ElementwiseKernel("add", size)
        runtime = get_runtime()

        # Pad if needed
        if size != kernel.size_padded:
            a_padded = np.zeros(kernel.size_padded, dtype=np.float16)
            b_padded = np.zeros(kernel.size_padded, dtype=np.float16)
            a_padded[:size] = a_flat.astype(np.float16)
            b_padded[:size] = b_flat.astype(np.float16)
        else:
            a_padded = a_flat.astype(np.float16)
            b_padded = b_flat.astype(np.float16)

        result_padded = runtime.execute(kernel.kernel_name, [a_padded, b_padded])
        result = result_padded[:size].astype(BF16_DTYPE)
        return result.reshape(orig_shape)
    else:
        raise RuntimeError("No NPU backend available")


def gelu_bf16(x: np.ndarray, use_npu: bool = True) -> np.ndarray:
    """
    GELU activation function.

    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

    The approximation used by many frameworks:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: Input tensor
        use_npu: Whether to attempt NPU acceleration

    Returns:
        GELU(x)
    """
    x = x.astype(BF16_DTYPE)
    orig_shape = x.shape
    x_flat = x.ravel()
    size = x_flat.size

    # Check if NPU acceleration is beneficial
    can_use_npu = (
        use_npu and
        (IRON_AVAILABLE or npu_available()) and
        size >= NPU_MIN_SIZE and
        x.flags['C_CONTIGUOUS']
    )

    if can_use_npu:
        try:
            return _gelu_bf16_npu(x_flat, orig_shape)
        except Exception:
            pass  # Fall through to CPU

    # CPU fallback: Tanh approximation (matches PyTorch default)
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    result = 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))
    return result.astype(BF16_DTYPE)


def _gelu_bf16_npu(x_flat: np.ndarray, orig_shape: tuple) -> np.ndarray:
    """NPU implementation of GELU."""
    size = x_flat.size

    if IRON_AVAILABLE:
        # Use modern IRON-based kernel
        from .npu.gelu_kernel import gelu_iron
        return gelu_iron(x_flat.reshape(orig_shape))
    elif npu_available():
        # Use legacy MLIR dialect approach
        kernel = ElementwiseKernel("gelu", size)
        runtime = get_runtime()

        # Pad if needed
        if size != kernel.size_padded:
            x_padded = np.zeros(kernel.size_padded, dtype=np.float16)
            x_padded[:size] = x_flat.astype(np.float16)
        else:
            x_padded = x_flat.astype(np.float16)

        result_padded = runtime.execute(kernel.kernel_name, [x_padded])
        result = result_padded[:size].astype(BF16_DTYPE)
        return result.reshape(orig_shape)
    else:
        raise RuntimeError("No NPU backend available")


def silu_bf16(x: np.ndarray) -> np.ndarray:
    """
    SiLU (Swish) activation: SiLU(x) = x * sigmoid(x)

    Args:
        x: Input tensor

    Returns:
        SiLU(x)
    """
    x = x.astype(BF16_DTYPE)
    return (x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))).astype(BF16_DTYPE)


def relu_bf16(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation: ReLU(x) = max(0, x)

    Args:
        x: Input tensor

    Returns:
        ReLU(x)
    """
    x = x.astype(BF16_DTYPE)
    return np.maximum(0, x).astype(BF16_DTYPE)


def sigmoid_bf16(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation: sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x: Input tensor

    Returns:
        sigmoid(x)
    """
    x = x.astype(BF16_DTYPE)
    # Clip to avoid overflow
    x_clipped = np.clip(x, -88, 88)
    return (1.0 / (1.0 + np.exp(-x_clipped))).astype(BF16_DTYPE)


def tanh_bf16(x: np.ndarray) -> np.ndarray:
    """
    Tanh activation.

    Args:
        x: Input tensor

    Returns:
        tanh(x)
    """
    x = x.astype(BF16_DTYPE)
    return np.tanh(x).astype(BF16_DTYPE)


def multiply_bf16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Element-wise multiplication: c = a * b

    Args:
        a: First operand
        b: Second operand

    Returns:
        c: Element-wise product
    """
    return (a.astype(BF16_DTYPE) * b.astype(BF16_DTYPE)).astype(BF16_DTYPE)


def divide_bf16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Element-wise division: c = a / b

    Args:
        a: Numerator
        b: Denominator

    Returns:
        c: Element-wise quotient
    """
    return (a.astype(BF16_DTYPE) / b.astype(BF16_DTYPE)).astype(BF16_DTYPE)
