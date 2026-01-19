"""
NPU kernels for Whisper inference using AMD IRON.

This module provides BF16 kernels that run on AMD Ryzen AI NPU.
Falls back to CPU (numpy) implementations when NPU is not available.

Two NPU backends are supported:
1. IRON (modern): High-level Python API using ObjectFifo/Worker/Runtime
2. MLIR-AIE (legacy): Lower-level MLIR dialect approach

The NPU programming model:
1. Kernels are defined using MLIR-AIE Python bindings
2. Compiled to xclbin (NPU bitstream) using aiecc.py
3. At runtime: load xclbin, DMA data to NPU, execute, DMA results back
"""

from .matmul import matmul_bf16, linear_bf16, batched_matmul_bf16, attention_scores_bf16
from .elementwise import add_bf16, gelu_bf16, silu_bf16, relu_bf16, sigmoid_bf16, tanh_bf16
from .layernorm import layernorm_bf16
from .softmax import softmax_bf16
from .conv1d import conv1d_bf16

# NPU runtime (legacy MLIR dialect)
try:
    from .npu_runtime import npu_available, get_runtime, NPURuntime, MLIR_AIE_AVAILABLE
except ImportError:
    MLIR_AIE_AVAILABLE = False

    def npu_available():
        return False

# IRON runtime (modern API)
try:
    from .npu import IRON_AVAILABLE
except ImportError:
    IRON_AVAILABLE = False


def get_npu_status() -> dict:
    """Get NPU availability status."""
    return {
        "iron_available": IRON_AVAILABLE,
        "mlir_aie_available": MLIR_AIE_AVAILABLE,
        "npu_available": npu_available(),
        "backend": (
            "iron" if IRON_AVAILABLE else
            "mlir_aie" if MLIR_AIE_AVAILABLE else
            "cpu"
        ),
    }


__all__ = [
    # Matrix operations
    "matmul_bf16",
    "linear_bf16",
    "batched_matmul_bf16",
    "attention_scores_bf16",
    # Elementwise operations
    "add_bf16",
    "gelu_bf16",
    "silu_bf16",
    "relu_bf16",
    "sigmoid_bf16",
    "tanh_bf16",
    # Normalization
    "layernorm_bf16",
    "softmax_bf16",
    # Convolution
    "conv1d_bf16",
    # Runtime
    "npu_available",
    "get_npu_status",
    "IRON_AVAILABLE",
    "MLIR_AIE_AVAILABLE",
]
