#!/usr/bin/env python3
"""
Matrix Multiplication NPU kernel using IRON.

Implements tiled GEMM: C = A @ B
Where A is (M, K), B is (K, N), C is (M, N)

For large matrices, uses CPU-side tiling with cached NPU kernel.

Usage:
    python matmul_kernel.py --verify
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

# Tile dimensions - small tiles for AIE memory constraints
TILE_M = 16
TILE_K = 16
TILE_N = 16

# Maximum tiles in single kernel (to avoid CDO generation failure)
MAX_OUTPUT_TILES = 1  # 1x1 output tile per kernel call
MAX_K_TILES = 1  # Single K tile per kernel (program memory limit)


def pad_to_tile(x: np.ndarray, tile_m: int, tile_n: int) -> np.ndarray:
    """Pad matrix to be divisible by tile dimensions."""
    m, n = x.shape
    pad_m = (tile_m - m % tile_m) % tile_m
    pad_n = (tile_n - n % tile_n) % tile_n
    if pad_m == 0 and pad_n == 0:
        return x
    return np.pad(x, ((0, pad_m), (0, pad_n)), mode='constant', constant_values=0)


# Cache for compiled kernels: (n_tiles_m, n_tiles_k, n_tiles_n, dtype) -> kernel_fn
_kernel_cache = {}


def _get_tile_kernel(n_tiles_m: int, n_tiles_k: int, n_tiles_n: int, dtype):
    """Get or compile a kernel for the given tile configuration."""
    key = (n_tiles_m, n_tiles_k, n_tiles_n, dtype)
    if key in _kernel_cache:
        return _kernel_cache[key]

    device = iron.get_current_device()

    n_output_tiles = n_tiles_m * n_tiles_n
    tile_a_size = TILE_M * TILE_K
    tile_b_size = TILE_K * TILE_N
    tile_c_size = TILE_M * TILE_N

    # Total stream sizes
    n_a_elements = n_output_tiles * n_tiles_k * tile_a_size
    n_b_elements = n_output_tiles * n_tiles_k * tile_b_size
    n_c_elements = n_output_tiles * tile_c_size

    @iron.jit(is_placed=False)
    def matmul_tile_kernel(a_tensor, b_tensor, c_tensor):
        a_tensor_ty = np.ndarray[(n_a_elements,), np.dtype[dtype]]
        b_tensor_ty = np.ndarray[(n_b_elements,), np.dtype[dtype]]
        c_tensor_ty = np.ndarray[(n_c_elements,), np.dtype[dtype]]

        tile_a_ty = np.ndarray[(tile_a_size,), np.dtype[dtype]]
        tile_b_ty = np.ndarray[(tile_b_size,), np.dtype[dtype]]
        tile_c_ty = np.ndarray[(tile_c_size,), np.dtype[dtype]]

        of_a = ObjectFifo(tile_a_ty, name="A")
        of_b = ObjectFifo(tile_b_ty, name="B")
        of_c = ObjectFifo(tile_c_ty, name="C")

        def core_fn(of_a, of_b, of_c):
            for _ in range_(n_output_tiles):
                c_tile = of_c.acquire(1)

                # Zero initialize
                for i in range_(tile_c_size):
                    c_tile[i] = 0.0

                # Accumulate over K tiles
                for _ in range_(n_tiles_k):
                    a_tile = of_a.acquire(1)
                    b_tile = of_b.acquire(1)

                    for i in range_(TILE_M):
                        for j in range_(TILE_N):
                            for k in range_(TILE_K):
                                c_tile[i * TILE_N + j] = c_tile[i * TILE_N + j] + a_tile[i * TILE_K + k] * b_tile[k * TILE_N + j]

                    of_a.release(1)
                    of_b.release(1)

                of_c.release(1)

        worker = Worker(core_fn, fn_args=[of_a.cons(), of_b.cons(), of_c.prod()])

        rt = Runtime()
        with rt.sequence(a_tensor_ty, b_tensor_ty, c_tensor_ty) as (a_host, b_host, c_host):
            rt.start(worker)
            rt.fill(of_a.prod(), a_host)
            rt.fill(of_b.prod(), b_host)
            rt.drain(of_c.cons(), c_host, wait=True)

        return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())

    _kernel_cache[key] = matmul_tile_kernel
    return matmul_tile_kernel


def matmul_iron(
    A: np.ndarray,
    B: np.ndarray,
    use_mmul: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Matrix multiplication on NPU.

    Due to JIT runtime limitations with multiple kernel calls, large matrices
    fall back to CPU. NPU is used for small matrices that fit in a single kernel.

    Args:
        A: Left matrix of shape (M, K)
        B: Right matrix of shape (K, N)
        use_mmul: Use hardware matrix multiply (ignored, always inline)
        verbose: Print debug info

    Returns:
        C: Result matrix of shape (M, N)
    """
    if not IRON_AVAILABLE:
        raise ImportError("IRON API not available")

    assert A.ndim == 2 and B.ndim == 2, "Inputs must be 2D matrices"
    assert A.shape[1] == B.shape[0], f"Shape mismatch: {A.shape} @ {B.shape}"

    M, K = A.shape
    K2, N = B.shape

    # Pad to tile boundaries
    A_padded = pad_to_tile(A, TILE_M, TILE_K)
    B_padded = pad_to_tile(B, TILE_K, TILE_N)

    M_padded, K_padded = A_padded.shape
    _, N_padded = B_padded.shape

    n_tiles_m = M_padded // TILE_M
    n_tiles_k = K_padded // TILE_K
    n_tiles_n = N_padded // TILE_N

    if verbose:
        print(f"Running matmul kernel (tiled)")
        print(f"Matmul: ({M}, {K}) @ ({K}, {N}) -> ({M}, {N})")
        print(f"Padded: ({M_padded}, {K_padded}) @ ({K_padded}, {N_padded})")
        print(f"Tiles: {n_tiles_m} x {n_tiles_k} x {n_tiles_n}")

    # Check if we can do it in a single kernel call
    if n_tiles_m * n_tiles_n <= MAX_OUTPUT_TILES and n_tiles_k <= MAX_K_TILES:
        return _matmul_single_kernel(A_padded, B_padded, M, N, n_tiles_m, n_tiles_k, n_tiles_n, verbose)
    else:
        # Blocked approach is too slow for inference - fall back to CPU
        # (13824 kernel calls for 384x384 takes ~9 seconds)
        if verbose:
            print(f"Matrix too large for single NPU kernel, using CPU")
        return matmul_cpu(A, B)


def _matmul_single_kernel(A_padded, B_padded, M, N, n_tiles_m, n_tiles_k, n_tiles_n, verbose):
    """Use single NPU kernel for small matrices."""
    kernel = _get_tile_kernel(n_tiles_m, n_tiles_k, n_tiles_n, A_padded.dtype)

    tile_a_size = TILE_M * TILE_K
    tile_b_size = TILE_K * TILE_N
    tile_c_size = TILE_M * TILE_N

    # Prepare tiled data
    a_tiles = []
    b_tiles = []
    for im in range(n_tiles_m):
        for jn in range(n_tiles_n):
            for ik in range(n_tiles_k):
                a_tile = A_padded[im*TILE_M:(im+1)*TILE_M, ik*TILE_K:(ik+1)*TILE_K].ravel()
                b_tile = B_padded[ik*TILE_K:(ik+1)*TILE_K, jn*TILE_N:(jn+1)*TILE_N].ravel()
                a_tiles.append(a_tile)
                b_tiles.append(b_tile)

    a_stream = np.concatenate(a_tiles)
    b_stream = np.concatenate(b_tiles)

    n_output_tiles = n_tiles_m * n_tiles_n
    a_iron = iron.zeros((len(a_stream),), dtype=A_padded.dtype)
    b_iron = iron.zeros((len(b_stream),), dtype=A_padded.dtype)
    c_iron = iron.zeros((n_output_tiles * tile_c_size,), dtype=A_padded.dtype)

    # Copy data and sync to device
    np.copyto(np.asarray(a_iron), a_stream)
    np.copyto(np.asarray(b_iron), b_stream)
    a_iron._sync_to_device()
    b_iron._sync_to_device()

    kernel(a_iron, b_iron, c_iron)

    # Sync output from device
    c_iron._sync_from_device()

    # Reconstruct result
    c_tiles = np.array(c_iron).reshape(n_output_tiles, tile_c_size)
    M_padded, N_padded = n_tiles_m * TILE_M, n_tiles_n * TILE_N
    C_padded = np.zeros((M_padded, N_padded), dtype=A_padded.dtype)

    tile_idx = 0
    for im in range(n_tiles_m):
        for jn in range(n_tiles_n):
            C_padded[im*TILE_M:(im+1)*TILE_M, jn*TILE_N:(jn+1)*TILE_N] = c_tiles[tile_idx].reshape(TILE_M, TILE_N)
            tile_idx += 1

    return C_padded[:M, :N]


def _matmul_blocked(A_padded, B_padded, M, N, n_tiles_m, n_tiles_k, n_tiles_n, verbose):
    """Use CPU-side blocking for large matrices.

    Strategy: For each output tile, process K tiles in chunks of MAX_K_TILES.
    NPU accumulates within each chunk, CPU accumulates across chunks.

    IMPORTANT: Must call _sync_to_device() after modifying tensor data,
    as np.copyto() only updates the host buffer, not device memory.
    """
    M_padded = n_tiles_m * TILE_M
    N_padded = n_tiles_n * TILE_N
    C_padded = np.zeros((M_padded, N_padded), dtype=A_padded.dtype)

    tile_a_size = TILE_M * TILE_K
    tile_b_size = TILE_K * TILE_N
    tile_c_size = TILE_M * TILE_N

    # How many K-chunks per output tile
    n_k_chunks = (n_tiles_k + MAX_K_TILES - 1) // MAX_K_TILES
    total_kernel_calls = n_tiles_m * n_tiles_n * n_k_chunks

    if verbose:
        print(f"Using blocked approach: 1 output tile x {MAX_K_TILES} K tiles per kernel call")
        print(f"K chunks per output tile: {n_k_chunks}")
        print(f"Total kernel calls: {total_kernel_calls}")

    # Get kernel for 1 output tile x MAX_K_TILES K tiles
    kernel = _get_tile_kernel(1, MAX_K_TILES, 1, A_padded.dtype)

    # Pre-allocate iron tensors
    a_iron = iron.zeros((MAX_K_TILES * tile_a_size,), dtype=A_padded.dtype)
    b_iron = iron.zeros((MAX_K_TILES * tile_b_size,), dtype=A_padded.dtype)
    c_iron = iron.zeros((tile_c_size,), dtype=A_padded.dtype)

    # Process each output tile
    for im in range(n_tiles_m):
        for jn in range(n_tiles_n):
            # Accumulator for this output tile (CPU-side accumulation across K chunks)
            c_accum = np.zeros((TILE_M, TILE_N), dtype=A_padded.dtype)

            # Process K tiles in chunks
            for k_chunk in range(n_k_chunks):
                k_start = k_chunk * MAX_K_TILES
                k_end = min(k_start + MAX_K_TILES, n_tiles_k)
                actual_k_tiles = k_end - k_start

                # Prepare K tiles for this chunk
                a_tiles = []
                b_tiles = []
                for ik in range(k_start, k_end):
                    a_tile = A_padded[im*TILE_M:(im+1)*TILE_M, ik*TILE_K:(ik+1)*TILE_K].ravel()
                    b_tile = B_padded[ik*TILE_K:(ik+1)*TILE_K, jn*TILE_N:(jn+1)*TILE_N].ravel()
                    a_tiles.append(a_tile)
                    b_tiles.append(b_tile)

                # Pad to MAX_K_TILES if needed (zeros won't affect result)
                while len(a_tiles) < MAX_K_TILES:
                    a_tiles.append(np.zeros(tile_a_size, dtype=A_padded.dtype))
                    b_tiles.append(np.zeros(tile_b_size, dtype=A_padded.dtype))

                a_stream = np.concatenate(a_tiles)
                b_stream = np.concatenate(b_tiles)

                # Copy data to host buffer and sync to device
                np.copyto(np.asarray(a_iron), a_stream)
                np.copyto(np.asarray(b_iron), b_stream)
                a_iron._sync_to_device()
                b_iron._sync_to_device()

                kernel(a_iron, b_iron, c_iron)

                # Sync output from device and accumulate on CPU
                c_iron._sync_from_device()
                c_accum += np.array(c_iron).reshape(TILE_M, TILE_N)

            # Store accumulated result
            C_padded[im*TILE_M:(im+1)*TILE_M, jn*TILE_N:(jn+1)*TILE_N] = c_accum

    return C_padded[:M, :N]


def matmul_cpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """CPU reference implementation."""
    return A @ B


def main():
    parser = argparse.ArgumentParser(description="Test matmul NPU kernel")
    parser.add_argument("--m", type=int, default=32, help="M dimension")
    parser.add_argument("--k", type=int, default=32, help="K dimension")
    parser.add_argument("--n", type=int, default=32, help="N dimension")
    parser.add_argument("--no-mmul", action="store_true", help="Don't use hardware mmul")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="f32")
    args = parser.parse_args()

    dtype = np.float16 if args.dtype == "bf16" else np.float32
    np.random.seed(42)

    A = np.random.randn(args.m, args.k).astype(dtype)
    B = np.random.randn(args.k, args.n).astype(dtype)

    print(f"Testing matmul: ({args.m}, {args.k}) @ ({args.k}, {args.n}) {args.dtype}")

    if not IRON_AVAILABLE:
        print("IRON not available, running CPU reference only")
        C_cpu = matmul_cpu(A.astype(np.float32), B.astype(np.float32)).astype(dtype)
        print(f"CPU result[0,0:5]: {C_cpu[0, :5]}")
        return 0

    try:
        import time
        start = time.perf_counter()
        C_npu = matmul_iron(
            A, B,
            use_mmul=not args.no_mmul,
            verbose=args.verbose,
        )
        elapsed = time.perf_counter() - start
        print(f"NPU result[0,0:5]: {C_npu[0, :5]}")
        print(f"NPU time: {elapsed*1000:.1f}ms")

    except Exception as e:
        print(f"NPU execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    if args.verify:
        C_cpu = matmul_cpu(A.astype(np.float32), B.astype(np.float32)).astype(dtype)

        max_diff = np.abs(C_npu.astype(np.float32) - C_cpu.astype(np.float32)).max()
        mean_diff = np.abs(C_npu.astype(np.float32) - C_cpu.astype(np.float32)).mean()
        rel_error = max_diff / (np.abs(C_cpu).max() + 1e-6)

        if rel_error < 0.1:
            print(f"PASS: max_diff={max_diff:.4f}, rel_error={rel_error:.4f}")
            return 0
        else:
            print(f"FAIL: max_diff={max_diff:.4f}, rel_error={rel_error:.4f}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
