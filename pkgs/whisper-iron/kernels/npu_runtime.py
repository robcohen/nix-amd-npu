"""
NPU Runtime for AMD Ryzen AI using MLIR-AIE/IRON.

This module provides the infrastructure for:
1. Compiling kernel definitions to xclbin
2. Loading xclbin onto the NPU
3. Executing kernels with data transfers

The IRON programming model:
- Define computation using Object FIFOs and tile cores
- Compile to xclbin using aiecc.py
- At runtime: load xclbin, DMA data to NPU, execute, DMA results back
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import numpy as np

# Check for IRON/mlir_aie availability
try:
    from aie.dialects.aie import (
        AIEDevice, tile, core, object_fifo, object_fifo_link,
        ObjectFifoPort, device, mem, shim_dma, end as aie_end
    )
    from aie.dialects.aiex import npu_dma_memcpy_nd, npu_sync, runtime_sequence
    from aie.dialects.scf import for_, yield_
    from aie.extras.dialects.ext import memref, arith
    from aie.extras.context import mlir_mod_ctx
    from aie.utils.xrt import setup_aie, execute as xrt_execute
    import aie.utils.trace as trace_utils
    MLIR_AIE_AVAILABLE = True
except ImportError:
    MLIR_AIE_AVAILABLE = False

# Check for XRT
try:
    import pyxrt
    XRT_AVAILABLE = True
except ImportError:
    XRT_AVAILABLE = False


class NPURuntime:
    """
    Manages NPU kernel compilation and execution.

    Usage:
        runtime = NPURuntime()
        runtime.load_kernel("matmul", matmul_xclbin_path, matmul_insts_path)
        result = runtime.execute("matmul", [A, B], output_shape, output_dtype)
    """

    def __init__(self, device_id: int = 0, cache_dir: Optional[str] = None):
        """
        Initialize NPU runtime.

        Args:
            device_id: XRT device ID (default: 0)
            cache_dir: Directory for caching compiled xclbin files
        """
        self.device_id = device_id
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "whisper-iron"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.loaded_kernels = {}
        self.device = None
        self.context = None

        if XRT_AVAILABLE:
            self._init_xrt()

    def _init_xrt(self):
        """Initialize XRT device and context."""
        try:
            self.device = pyxrt.device(self.device_id)
            print(f"NPU device initialized: {self.device.get_info(pyxrt.xrt_info_device.name)}")
        except Exception as e:
            print(f"Warning: Could not initialize NPU device: {e}")
            self.device = None

    @property
    def is_available(self) -> bool:
        """Check if NPU is available."""
        return self.device is not None and MLIR_AIE_AVAILABLE

    def compile_kernel(
        self,
        name: str,
        mlir_source: str,
        force_recompile: bool = False,
    ) -> Tuple[Path, Path]:
        """
        Compile MLIR source to xclbin.

        Args:
            name: Kernel name
            mlir_source: MLIR source code
            force_recompile: Force recompilation even if cached

        Returns:
            Tuple of (xclbin_path, insts_path)
        """
        xclbin_path = self.cache_dir / f"{name}.xclbin"
        insts_path = self.cache_dir / f"{name}.insts.txt"

        if not force_recompile and xclbin_path.exists() and insts_path.exists():
            return xclbin_path, insts_path

        # Write MLIR source
        mlir_path = self.cache_dir / f"{name}.mlir"
        mlir_path.write_text(mlir_source)

        # Compile using aiecc.py
        cmd = [
            "aiecc.py",
            "--aie-generate-cdo",
            "--aie-generate-npu",
            "--no-aiesim",
            "--no-xchesscc",
            "--no-xbridge",
            f"--npu-insts-name={insts_path}",
            str(mlir_path),
            "-o", str(xclbin_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Compiled kernel '{name}' to {xclbin_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Kernel compilation failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("aiecc.py not found. Make sure IRON is installed and in PATH.")

        return xclbin_path, insts_path

    def load_kernel(
        self,
        name: str,
        xclbin_path: Path,
        insts_path: Path,
        input_shapes: List[Tuple[int, ...]],
        input_dtypes: List[np.dtype],
        output_shape: Tuple[int, ...],
        output_dtype: np.dtype,
    ):
        """
        Load a compiled kernel.

        Args:
            name: Kernel name
            xclbin_path: Path to xclbin file
            insts_path: Path to NPU instructions file
            input_shapes: Shapes of input tensors
            input_dtypes: Data types of input tensors
            output_shape: Shape of output tensor
            output_dtype: Data type of output tensor
        """
        if not self.is_available:
            print(f"Warning: NPU not available, kernel '{name}' will use CPU fallback")
            return

        app = setup_aie(
            str(xclbin_path),
            str(insts_path),
            input_shapes[0] if input_shapes else (1,),
            input_dtypes[0] if input_dtypes else np.float32,
            output_shape,
            output_dtype,
        )

        self.loaded_kernels[name] = {
            "app": app,
            "input_shapes": input_shapes,
            "input_dtypes": input_dtypes,
            "output_shape": output_shape,
            "output_dtype": output_dtype,
        }

    def execute(
        self,
        name: str,
        inputs: List[np.ndarray],
        output: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Execute a loaded kernel.

        Args:
            name: Kernel name
            inputs: List of input arrays
            output: Optional pre-allocated output array

        Returns:
            Output array
        """
        if name not in self.loaded_kernels:
            raise KeyError(f"Kernel '{name}' not loaded")

        kernel = self.loaded_kernels[name]

        if output is None:
            output = np.zeros(kernel["output_shape"], dtype=kernel["output_dtype"])

        # Execute on NPU
        result = xrt_execute(kernel["app"], inputs[0], output)

        return result


# Global runtime instance
_runtime: Optional[NPURuntime] = None


def get_runtime() -> NPURuntime:
    """Get or create the global NPU runtime."""
    global _runtime
    if _runtime is None:
        _runtime = NPURuntime()
    return _runtime


def npu_available() -> bool:
    """Check if NPU is available for computation."""
    return get_runtime().is_available


# ============================================================================
# Kernel Definition Helpers
# ============================================================================

def generate_matmul_mlir(M: int, K: int, N: int, dtype: str = "bf16") -> str:
    """
    Generate MLIR for matrix multiplication kernel.

    Args:
        M: Rows of A and C
        K: Columns of A, rows of B
        N: Columns of B and C
        dtype: Data type ("bf16" or "f32")

    Returns:
        MLIR source code
    """
    if not MLIR_AIE_AVAILABLE:
        raise RuntimeError("MLIR-AIE not available")

    # Map dtype to MLIR type
    mlir_type = "bf16" if dtype == "bf16" else "f32"

    # Tile sizes for AIE (must fit in 16KB tile memory)
    # For bf16: 16KB / 2 bytes = 8192 elements
    # Typical tile: 64x64 = 4096 elements per matrix
    tile_m, tile_k, tile_n = 64, 64, 64

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu1_1col)
        def matmul_device():
            # Memory types
            a_tile_ty = T.memref(tile_m, tile_k, T.bf16() if dtype == "bf16" else T.f32())
            b_tile_ty = T.memref(tile_k, tile_n, T.bf16() if dtype == "bf16" else T.f32())
            c_tile_ty = T.memref(tile_m, tile_n, T.bf16() if dtype == "bf16" else T.f32())

            # Tiles
            shim_tile = tile(0, 0)
            mem_tile = tile(0, 1)
            compute_tile = tile(0, 2)

            # Object FIFOs for data movement
            of_a = object_fifo("A", shim_tile, compute_tile, 2, a_tile_ty)
            of_b = object_fifo("B", shim_tile, compute_tile, 2, b_tile_ty)
            of_c = object_fifo("C", compute_tile, shim_tile, 2, c_tile_ty)

            # Compute core
            @core(compute_tile)
            def core_body():
                for _ in for_(M // tile_m):
                    for _ in for_(N // tile_n):
                        # Acquire output tile (zero initialize)
                        c_buf = of_c.acquire(ObjectFifoPort.Produce, 1)

                        # Initialize C to zero
                        for i in for_(tile_m):
                            for j in for_(tile_n):
                                zero = arith.constant(T.bf16() if dtype == "bf16" else T.f32(), 0.0)
                                memref.store(zero, c_buf, [i, j])
                                yield_()
                            yield_()

                        # Accumulate over K dimension
                        for _ in for_(K // tile_k):
                            a_buf = of_a.acquire(ObjectFifoPort.Consume, 1)
                            b_buf = of_b.acquire(ObjectFifoPort.Consume, 1)

                            # C += A @ B (naive triple loop)
                            for i in for_(tile_m):
                                for j in for_(tile_n):
                                    for k in for_(tile_k):
                                        a_val = memref.load(a_buf, [i, k])
                                        b_val = memref.load(b_buf, [k, j])
                                        c_val = memref.load(c_buf, [i, j])
                                        prod = arith.mulf(a_val, b_val)
                                        new_c = arith.addf(c_val, prod)
                                        memref.store(new_c, c_buf, [i, j])
                                        yield_()
                                    yield_()
                                yield_()

                            of_a.release(ObjectFifoPort.Consume, 1)
                            of_b.release(ObjectFifoPort.Consume, 1)
                            yield_()

                        of_c.release(ObjectFifoPort.Produce, 1)
                        yield_()
                    yield_()

            # Runtime sequence for host-device data movement
            @runtime_sequence(
                T.memref(M, K, T.bf16() if dtype == "bf16" else T.f32()),
                T.memref(K, N, T.bf16() if dtype == "bf16" else T.f32()),
                T.memref(M, N, T.bf16() if dtype == "bf16" else T.f32()),
            )
            def sequence(A, B, C):
                # Transfer A tiles
                for im in range(M // tile_m):
                    for ik in range(K // tile_k):
                        npu_dma_memcpy_nd(
                            metadata="A",
                            bd_id=0,
                            mem=A,
                            offsets=[0, 0, im * tile_m, ik * tile_k],
                            sizes=[1, 1, tile_m, tile_k],
                        )

                # Transfer B tiles
                for ik in range(K // tile_k):
                    for jn in range(N // tile_n):
                        npu_dma_memcpy_nd(
                            metadata="B",
                            bd_id=1,
                            mem=B,
                            offsets=[0, 0, ik * tile_k, jn * tile_n],
                            sizes=[1, 1, tile_k, tile_n],
                        )

                # Receive C tiles
                for im in range(M // tile_m):
                    for jn in range(N // tile_n):
                        npu_dma_memcpy_nd(
                            metadata="C",
                            bd_id=2,
                            mem=C,
                            offsets=[0, 0, im * tile_m, jn * tile_n],
                            sizes=[1, 1, tile_m, tile_n],
                        )

                npu_sync(column=0, row=0, direction=0, channel=0)

        return str(ctx.module)


def generate_elementwise_mlir(
    size: int,
    op: str,  # "gelu", "add", "relu", etc.
    dtype: str = "bf16",
) -> str:
    """
    Generate MLIR for elementwise operations.

    Args:
        size: Number of elements
        op: Operation type
        dtype: Data type

    Returns:
        MLIR source code
    """
    if not MLIR_AIE_AVAILABLE:
        raise RuntimeError("MLIR-AIE not available")

    tile_size = min(size, 1024)  # Process in chunks

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu1_1col)
        def elementwise_device():
            elem_ty = T.bf16() if dtype == "bf16" else T.f32()
            buf_ty = T.memref(tile_size, elem_ty)

            shim_tile = tile(0, 0)
            compute_tile = tile(0, 2)

            of_in = object_fifo("in", shim_tile, compute_tile, 2, buf_ty)
            of_out = object_fifo("out", compute_tile, shim_tile, 2, buf_ty)

            @core(compute_tile)
            def core_body():
                for _ in for_(size // tile_size):
                    in_buf = of_in.acquire(ObjectFifoPort.Consume, 1)
                    out_buf = of_out.acquire(ObjectFifoPort.Produce, 1)

                    for i in for_(tile_size):
                        x = memref.load(in_buf, [i])

                        if op == "gelu":
                            # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                            # Simplified for NPU: use polynomial approximation
                            x2 = arith.mulf(x, x)
                            x3 = arith.mulf(x2, x)
                            coef = arith.constant(elem_ty, 0.044715)
                            inner = arith.addf(x, arith.mulf(coef, x3))
                            # tanh approximation or use built-in
                            half = arith.constant(elem_ty, 0.5)
                            one = arith.constant(elem_ty, 1.0)
                            result = arith.mulf(half, arith.mulf(x, arith.addf(one, inner)))
                        elif op == "relu":
                            zero = arith.constant(elem_ty, 0.0)
                            result = arith.maximumf(x, zero)
                        elif op == "sigmoid":
                            # sigmoid(x) = 1 / (1 + exp(-x))
                            one = arith.constant(elem_ty, 1.0)
                            neg_x = arith.negf(x)
                            exp_neg_x = arith.exp(neg_x)
                            denom = arith.addf(one, exp_neg_x)
                            result = arith.divf(one, denom)
                        else:
                            result = x  # passthrough

                        memref.store(result, out_buf, [i])
                        yield_()

                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                    yield_()

            @runtime_sequence(buf_ty, buf_ty)
            def sequence(input_tensor, output_tensor):
                for chunk in range(size // tile_size):
                    npu_dma_memcpy_nd(
                        metadata="in",
                        bd_id=0,
                        mem=input_tensor,
                        offsets=[0, 0, 0, chunk * tile_size],
                        sizes=[1, 1, 1, tile_size],
                    )
                    npu_dma_memcpy_nd(
                        metadata="out",
                        bd_id=1,
                        mem=output_tensor,
                        offsets=[0, 0, 0, chunk * tile_size],
                        sizes=[1, 1, 1, tile_size],
                    )
                npu_sync(column=0, row=0, direction=0, channel=0)

        return str(ctx.module)
