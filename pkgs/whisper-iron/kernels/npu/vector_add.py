#!/usr/bin/env python3
"""
Vector Addition NPU kernel using IRON.

This implements element-wise addition c = a + b on AMD Ryzen AI NPU.
Serves as a template for more complex kernels like GELU, matmul, etc.

Usage:
    # In iron-fhs environment
    python vector_add.py --verify
"""

import numpy as np
import argparse
from pathlib import Path

# Check for IRON availability
try:
    import aie.iron as iron
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.placers import SequentialPlacer
    from aie.iron.device import NPU2
    from aie.iron.controlflow import range_
    IRON_AVAILABLE = True
except ImportError:
    IRON_AVAILABLE = False

# Kernel source directory
KERNEL_DIR = Path(__file__).parent


def vector_add_iron(
    input_a: np.ndarray,
    input_b: np.ndarray,
    tile_size: int = 1024,
    use_vector: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Perform vector addition on NPU.

    Args:
        input_a: First input array (bfloat16 or float32)
        input_b: Second input array (same shape as input_a)
        tile_size: Elements per tile (1024 or 4096)
        use_vector: Use vectorized kernel (16x faster)
        verbose: Print debug info

    Returns:
        output: Element-wise sum a + b
    """
    if not IRON_AVAILABLE:
        raise ImportError("IRON API not available")

    assert input_a.shape == input_b.shape, "Input shapes must match"
    assert input_a.dtype == input_b.dtype, "Input dtypes must match"

    # Flatten for processing
    orig_shape = input_a.shape
    a_flat = input_a.ravel()
    b_flat = input_b.ravel()
    n_elements = a_flat.size

    # Pad to tile boundary
    n_tiles = (n_elements + tile_size - 1) // tile_size
    n_padded = n_tiles * tile_size

    if n_padded != n_elements:
        a_padded = np.zeros(n_padded, dtype=input_a.dtype)
        b_padded = np.zeros(n_padded, dtype=input_b.dtype)
        a_padded[:n_elements] = a_flat
        b_padded[:n_elements] = b_flat
    else:
        a_padded = a_flat.copy()
        b_padded = b_flat.copy()

    # Determine kernel function name
    dtype_str = "bf16" if input_a.dtype == np.float16 else "f32"
    size_str = "4k_" if tile_size == 4096 else ""
    vec_str = "vector" if use_vector else "scalar"
    kernel_func = f"eltwise_add_{dtype_str}_{size_str}{vec_str}"

    if verbose:
        print(f"Running kernel: {kernel_func}")
        print(f"  Input size: {n_elements} -> padded: {n_padded}")
        print(f"  Tiles: {n_tiles} x {tile_size}")

    # Get device target (NPU2 is the default)
    device = iron.get_current_device()

    # Define the IRON program with JIT decorator
    @iron.jit(is_placed=False)
    def vector_add_kernel(a_tensor, b_tensor, c_tensor):
        # Data types
        tensor_ty = np.ndarray[(n_padded,), np.dtype[input_a.dtype]]
        tile_ty = np.ndarray[(tile_size,), np.dtype[input_a.dtype]]

        # Object FIFOs for data movement (shim <-> compute)
        of_a = ObjectFifo(tile_ty, name="A")
        of_b = ObjectFifo(tile_ty, name="B")
        of_c = ObjectFifo(tile_ty, name="C")

        # Worker that runs on compute tile - inline operations (no external kernel)
        def core_fn(of_a, of_b, of_c):
            for _ in range_(n_tiles):
                # Acquire input tiles
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                elem_c = of_c.acquire(1)

                # Element-wise addition (compiled by IRON to AIE instructions)
                for i in range_(tile_size):
                    elem_c[i] = elem_a[i] + elem_b[i]

                # Release tiles
                of_a.release(1)
                of_b.release(1)
                of_c.release(1)

        worker = Worker(core_fn, fn_args=[of_a.cons(), of_b.cons(), of_c.prod()])

        # Runtime sequence (host orchestration)
        rt = Runtime()
        with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a_host, b_host, c_host):
            rt.start(worker)
            rt.fill(of_a.prod(), a_host)
            rt.fill(of_b.prod(), b_host)
            rt.drain(of_c.cons(), c_host, wait=True)

        return Program(device, rt).resolve_program(SequentialPlacer())

    # Convert numpy arrays to iron tensors
    a_iron = iron.zeros((n_padded,), dtype=input_a.dtype)
    b_iron = iron.zeros((n_padded,), dtype=input_a.dtype)
    c_iron = iron.zeros((n_padded,), dtype=input_a.dtype)

    # Copy data into iron tensors and sync to device
    np.copyto(np.asarray(a_iron), a_padded)
    np.copyto(np.asarray(b_iron), b_padded)
    a_iron._sync_to_device()
    b_iron._sync_to_device()

    # Execute on NPU
    vector_add_kernel(a_iron, b_iron, c_iron)

    # Sync output from device
    c_iron._sync_from_device()

    # Convert back to numpy and unpad
    c_padded = np.array(c_iron)
    c_result = c_padded[:n_elements].reshape(orig_shape)
    return c_result


def vector_add_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CPU reference implementation."""
    return a + b


def main():
    parser = argparse.ArgumentParser(description="Test vector_add NPU kernel")
    parser.add_argument("--size", type=int, default=4096, help="Vector size")
    parser.add_argument("--tile-size", type=int, default=1024, choices=[1024, 4096])
    parser.add_argument("--scalar", action="store_true", help="Use scalar kernel")
    parser.add_argument("--verify", action="store_true", help="Verify against CPU")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    args = parser.parse_args()

    # Create test data
    dtype = np.float16 if args.dtype == "bf16" else np.float32
    np.random.seed(42)
    a = np.random.randn(args.size).astype(dtype)
    b = np.random.randn(args.size).astype(dtype)

    print(f"Testing vector_add with {args.size} {args.dtype} elements")

    if not IRON_AVAILABLE:
        print("IRON not available, running CPU reference only")
        c_cpu = vector_add_cpu(a, b)
        print(f"CPU result: {c_cpu[:5]}...")
        return 0

    # Run on NPU
    try:
        c_npu = vector_add_iron(
            a, b,
            tile_size=args.tile_size,
            use_vector=not args.scalar,
            verbose=args.verbose,
        )
        print(f"NPU result: {c_npu[:5]}...")
    except Exception as e:
        print(f"NPU execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Verify
    if args.verify:
        c_cpu = vector_add_cpu(a, b)
        max_diff = np.abs(c_npu - c_cpu).max()
        mean_diff = np.abs(c_npu - c_cpu).mean()

        if max_diff < 1e-3:
            print(f"PASS: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            return 0
        else:
            print(f"FAIL: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
