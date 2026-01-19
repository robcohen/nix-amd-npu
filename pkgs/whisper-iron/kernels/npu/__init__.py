"""
NPU kernels using MLIR-AIE/IRON.

This package contains actual NPU kernel implementations that run on
AMD Ryzen AI NPU (AIE2 tiles). Each kernel consists of:
1. C++ kernel source (.cc) - runs on AIE compute tiles
2. Python IRON wrapper (.py) - defines data flow and orchestration

Available kernels:
- vector_add: Element-wise addition (add.cc, vector_add.py)
- gelu: GELU activation (gelu.cc, gelu_kernel.py)
- softmax: Row-wise softmax (softmax.cc, softmax_kernel.py)
- layernorm: Layer normalization (layernorm.cc, layernorm_kernel.py)
- matmul: Tiled matrix multiplication (matmul.cc, matmul_kernel.py)
- conv1d: 1D convolution for audio (conv1d.cc, conv1d_kernel.py)

To use NPU kernels, you must be in the iron-fhs environment with
mlir_aie, llvm-aie, and eudsl-python-extras installed.
"""

from pathlib import Path

# Path to this directory (contains .cc kernel sources)
NPU_KERNELS_DIR = Path(__file__).parent

# Track availability
IRON_AVAILABLE = False
try:
    from aie.iron import ObjectFifo, Worker, Runtime, Kernel
    from aie.iron.device import NPU2
    IRON_AVAILABLE = True
except ImportError:
    pass


def check_iron():
    """Check if IRON API is available."""
    if not IRON_AVAILABLE:
        raise ImportError(
            "IRON API not available. Please run in iron-fhs environment with:\n"
            "  EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX=aie pip install eudsl-python-extras==0.1.0.20251215.1800+3c7ac1b -f https://llvm.github.io/eudsl\n"
            "  pip install mlir_aie llvm-aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels"
        )


def get_kernel_status() -> dict:
    """Get status of all NPU kernels."""
    kernels = {
        "add": {"cc": "add.cc", "wrapper": "vector_add.py"},
        "gelu": {"cc": "gelu.cc", "wrapper": "gelu_kernel.py"},
        "softmax": {"cc": "softmax.cc", "wrapper": "softmax_kernel.py"},
        "layernorm": {"cc": "layernorm.cc", "wrapper": "layernorm_kernel.py"},
        "matmul": {"cc": "matmul.cc", "wrapper": "matmul_kernel.py"},
        "conv1d": {"cc": "conv1d.cc", "wrapper": "conv1d_kernel.py"},
    }

    status = {}
    for name, files in kernels.items():
        cc_path = NPU_KERNELS_DIR / files["cc"]
        obj_path = NPU_KERNELS_DIR / (files["cc"] + ".o")
        wrapper_path = NPU_KERNELS_DIR / files["wrapper"]

        status[name] = {
            "source_exists": cc_path.exists(),
            "compiled": obj_path.exists(),
            "wrapper_exists": wrapper_path.exists(),
        }

    return status
