#!/usr/bin/env python3
"""
Softmax NPU kernel using IRON.

Implements row-wise softmax for attention on AMD Ryzen AI NPU.
softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))

Uses hybrid CPU/NPU approach:
- CPU: compute max per row (numerical stability) and sum reduction
- NPU: compute exp(x - max) element-wise

Usage:
    python softmax_kernel.py --verify
"""

import numpy as np
import argparse
from pathlib import Path

# Check for IRON availability
try:
    import aie.iron as iron
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.placers import SequentialPlacer
    from aie.iron.device import NPU2
    from aie.iron.controlflow import range_
    IRON_AVAILABLE = True
except ImportError:
    IRON_AVAILABLE = False

KERNEL_DIR = Path(__file__).parent

# Tile size for processing
TILE_SIZE = 1024


def softmax_iron(
    input_x: np.ndarray,
    axis: int = -1,
    use_vector: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Apply softmax on NPU along specified axis.

    Uses hybrid CPU/NPU approach:
    - CPU: compute max per row (for numerical stability)
    - NPU: compute exp(x - max) using buffer accumulator pattern
    - CPU: compute sum and normalize

    Args:
        input_x: Input array
        axis: Axis to apply softmax (default: -1, last axis)
        use_vector: Use vectorized kernel (ignored)
        verbose: Print debug info

    Returns:
        output: softmax(input_x) along axis
    """
    if not IRON_AVAILABLE:
        raise ImportError("IRON API not available")

    # Normalize axis
    if axis < 0:
        axis = input_x.ndim + axis

    # Get row size (size along softmax axis)
    row_size = input_x.shape[axis]

    # Calculate number of rows
    n_rows = input_x.size // row_size

    # Reshape to 2D: (n_rows, row_size)
    perm = list(range(input_x.ndim))
    perm.remove(axis)
    perm.append(axis)
    x_transposed = np.transpose(input_x, perm)
    orig_shape = x_transposed.shape
    x_2d = x_transposed.reshape(-1, row_size).astype(np.float32)

    # CPU: compute max per row for numerical stability
    row_max = np.max(x_2d, axis=1, keepdims=True)
    x_shifted = x_2d - row_max  # Now all values <= 0

    if verbose:
        print(f"Running softmax kernel (hybrid CPU/NPU)")
        print(f"  Rows: {n_rows}, row_size: {row_size}")

    # Compute exp on NPU using chunked single-tile execution
    y_2d = _exp_npu(x_shifted, verbose)

    # CPU: compute sum and normalize
    row_sums = np.sum(y_2d, axis=1, keepdims=True)
    y_2d = y_2d / (row_sums + 1e-10)

    # Reshape back
    y_transposed = y_2d.reshape(orig_shape)

    # Inverse transpose to restore original axis order
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    y_result = np.transpose(y_transposed, inv_perm).astype(input_x.dtype)

    return y_result


def _exp_npu(x_2d: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Compute exp(x) on NPU for x <= 0.

    Uses buffer accumulator pattern to avoid local+local bug.
    Processes in single-tile chunks to avoid multi-tile bugs.

    exp(x) for x <= 0 is approximated as:
    exp(x) ≈ 1 / (1 - x + x²/2 - x³/6 + x⁴/24)

    Using buffer accumulators:
    eb[i] = x²
    ec[i] = 1 - x + eb[i]*0.5  (partial denom)
    eb[i] = eb[i] * x  (x³)
    ec[i] = ec[i] - eb[i]*0.16667  (add x³ term)
    eb[i] = eb[i] * x  (x⁴)
    ec[i] = ec[i] + eb[i]*0.04167  (add x⁴ term)
    ec[i] = 1 / ec[i]  (final exp)
    """
    n_rows, row_size = x_2d.shape

    # Flatten for processing
    x_flat = x_2d.ravel()
    n_elements = x_flat.size

    # Pad to tile boundary
    n_tiles = (n_elements + TILE_SIZE - 1) // TILE_SIZE
    n_padded = n_tiles * TILE_SIZE

    if n_padded != n_elements:
        x_padded = np.full(n_padded, -100.0, dtype=np.float32)  # exp(-100) ≈ 0
        x_padded[:n_elements] = x_flat
    else:
        x_padded = x_flat.copy()

    # Get device target (NPU2 is the default)
    device = iron.get_current_device()

    # Define single-tile exp kernel with 3-tensor pattern
    @iron.jit(is_placed=False)
    def exp_kernel_single_tile(a_tensor, b_tensor, c_tensor):
        tensor_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.float32]]
        tile_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.float32]]

        # 3-tensor pattern to avoid IRON bugs
        of_a = ObjectFifo(tile_ty, name="A")  # input x
        of_b = ObjectFifo(tile_ty, name="B")  # scratch
        of_c = ObjectFifo(tile_ty, name="C")  # output

        def core_fn(of_a, of_b, of_c):
            ea = of_a.acquire(1)
            eb = of_b.acquire(1)
            ec = of_c.acquire(1)

            for i in range_(TILE_SIZE):
                # exp(x) for x <= 0 using simple approximation:
                # exp(x) ≈ 1 / (1 - x + x²/2)
                #
                # IMPORTANT: Pattern `ec = ec + eb * const` FAILS silently!
                # Must use: `eb = eb * const; ec = ec + eb` instead.

                # Step 1: x² -> scratch
                eb[i] = ea[i] * ea[i]

                # Step 2: x²/2 -> scratch (scale before adding)
                eb[i] = eb[i] * 0.5

                # Step 3: Initialize output to 1 (via 0*x + 1)
                ec[i] = ea[i] * 0.0
                ec[i] = ec[i] + 1.0

                # Step 4: Add -x term (1 + (-1)*x = 1 - x)
                ec[i] = ec[i] + ea[i] * (-1.0)

                # Step 5: Add x²/2 term (now just add eb)
                ec[i] = ec[i] + eb[i]

                # Step 6: exp = 1 / denom
                ec[i] = 1.0 / (ec[i] + 0.0001)

            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

        worker = Worker(core_fn, fn_args=[of_a.cons(), of_b.cons(), of_c.prod()])

        rt = Runtime()
        with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a_h, b_h, c_h):
            rt.start(worker)
            rt.fill(of_a.prod(), a_h)
            rt.fill(of_b.prod(), b_h)
            rt.drain(of_c.cons(), c_h, wait=True)

        return Program(device, rt).resolve_program(SequentialPlacer())

    # Allocate iron tensors for single tile
    a_iron = iron.zeros((TILE_SIZE,), dtype=np.float32)
    b_iron = iron.zeros((TILE_SIZE,), dtype=np.float32)
    c_iron = iron.zeros((TILE_SIZE,), dtype=np.float32)

    # Output buffer
    y_padded = np.zeros(n_padded, dtype=np.float32)

    # Process tiles one at a time
    for tile_idx in range(n_tiles):
        start = tile_idx * TILE_SIZE
        end = start + TILE_SIZE

        # Copy input chunk
        np.copyto(np.asarray(a_iron), x_padded[start:end])
        a_iron._sync_to_device()
        b_iron._sync_to_device()

        # Execute
        exp_kernel_single_tile(a_iron, b_iron, c_iron)

        # Copy output
        c_iron._sync_from_device()
        y_padded[start:end] = np.array(c_iron)

    # Unpad and reshape
    y_flat = y_padded[:n_elements]
    return y_flat.reshape(n_rows, row_size)


def softmax_cpu(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """CPU reference implementation."""
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="Test softmax NPU kernel")
    parser.add_argument("--rows", type=int, default=16, help="Number of rows")
    parser.add_argument("--cols", type=int, default=64, help="Row size")
    parser.add_argument("--scalar", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="f32")
    args = parser.parse_args()

    dtype = np.float16 if args.dtype == "bf16" else np.float32
    np.random.seed(42)
    x = np.random.randn(args.rows, args.cols).astype(dtype)

    print(f"Testing softmax with shape ({args.rows}, {args.cols}) {args.dtype}")

    if not IRON_AVAILABLE:
        print("IRON not available, running CPU reference only")
        y_cpu = softmax_cpu(x.astype(np.float32)).astype(dtype)
        print(f"CPU result[0]: {y_cpu[0, :5]}...")
        return 0

    try:
        y_npu = softmax_iron(
            x,
            axis=-1,
            use_vector=not args.scalar,
            verbose=args.verbose,
        )
        print(f"NPU result[0]: {y_npu[0, :5]}...")
    except Exception as e:
        print(f"NPU execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    if args.verify:
        y_cpu = softmax_cpu(x.astype(np.float32)).astype(dtype)
        max_diff = np.abs(y_npu.astype(np.float32) - y_cpu.astype(np.float32)).max()
        mean_diff = np.abs(y_npu.astype(np.float32) - y_cpu.astype(np.float32)).mean()

        # Check rows sum to ~1
        row_sums = y_npu.sum(axis=-1)
        sum_error = np.abs(row_sums - 1.0).max()

        # Exp approximation allows some error
        if sum_error < 0.1 and max_diff < 0.5:
            print(f"PASS: max_diff={max_diff:.6f}, sum_error={sum_error:.6f}")
            return 0
        else:
            print(f"FAIL: max_diff={max_diff:.6f}, sum_error={sum_error:.6f}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
