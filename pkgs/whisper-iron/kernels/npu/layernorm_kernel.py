#!/usr/bin/env python3
"""
Layer Normalization NPU kernel using IRON.

Implements: y = (x - mean) / sqrt(var + eps) * gamma + beta

Usage:
    python layernorm_kernel.py --verify
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

# Supported normalization sizes
SUPPORTED_SIZES = [64, 128, 256, 384, 512, 768, 1024, 1536]


def get_padded_size(size: int) -> int:
    """Get the smallest supported size >= input size."""
    for s in SUPPORTED_SIZES:
        if s >= size:
            return s
    raise ValueError(f"Norm size {size} too large, max is {SUPPORTED_SIZES[-1]}")


def layernorm_iron(
    input_x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    axis: int = -1,
    eps: float = 1e-5,
    use_vector: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Apply layer normalization using hybrid CPU/NPU approach.

    CPU computes mean/variance (reductions), NPU runs affine transform.
    This avoids program memory overflow from complex reduction loops.

    Args:
        input_x: Input array of shape (..., norm_size)
        gamma: Scale parameter of shape (norm_size,)
        beta: Bias parameter of shape (norm_size,)
        axis: Axis to normalize (default: -1)
        eps: Epsilon for numerical stability
        use_vector: Use vectorized kernel (ignored)
        verbose: Print debug info

    Returns:
        output: Normalized array
    """
    if not IRON_AVAILABLE:
        raise ImportError("IRON API not available")

    # Normalize axis
    if axis < 0:
        axis = input_x.ndim + axis

    # Get normalization size
    norm_size = input_x.shape[axis]
    padded_size = get_padded_size(norm_size)

    assert gamma.shape == (norm_size,), f"gamma shape mismatch"
    assert beta.shape == (norm_size,), f"beta shape mismatch"

    # === CPU: Compute mean and variance (reductions) ===
    mean = np.mean(input_x, axis=axis, keepdims=True)
    var = np.var(input_x, axis=axis, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)

    # Pre-normalize on CPU: x_norm = (x - mean) * inv_std
    x_norm = (input_x - mean) * inv_std

    if verbose:
        print(f"Running layernorm kernel (hybrid CPU/NPU)")
        print(f"  CPU: computed mean/var, pre-normalized")
        print(f"  NPU: affine transform y = x_norm * gamma + beta")

    # Calculate number of vectors to normalize
    n_vectors = input_x.size // norm_size

    # Reshape to 2D: (n_vectors, norm_size)
    perm = list(range(input_x.ndim))
    perm.remove(axis)
    perm.append(axis)
    x_transposed = np.transpose(x_norm, perm)
    orig_shape = x_transposed.shape
    x_2d = x_transposed.reshape(-1, norm_size)

    # Pad if needed
    if padded_size != norm_size:
        x_padded = np.zeros((n_vectors, padded_size), dtype=input_x.dtype)
        x_padded[:, :norm_size] = x_2d
        gamma_padded = np.zeros(padded_size, dtype=gamma.dtype)
        gamma_padded[:norm_size] = gamma
        beta_padded = np.zeros(padded_size, dtype=beta.dtype)
        beta_padded[:norm_size] = beta
    else:
        x_padded = x_2d.copy()
        gamma_padded = gamma.copy()
        beta_padded = beta.copy()

    # Flatten for IRON processing
    n_elements = n_vectors * padded_size
    x_flat = x_padded.ravel()

    if verbose:
        print(f"  Vectors: {n_vectors}, norm_size: {norm_size} -> padded: {padded_size}")

    # Get device target (NPU2 is the default)
    device = iron.get_current_device()

    # Combine gamma and beta into interleaved buffer: [g0, b0, g1, b1, ...]
    params_interleaved = np.zeros(2 * padded_size, dtype=input_x.dtype)
    params_interleaved[0::2] = gamma_padded
    params_interleaved[1::2] = beta_padded

    @iron.jit(is_placed=False)
    def layernorm_kernel_jit(x_tensor, params_tensor, y_tensor):
        tensor_ty = np.ndarray[(n_elements,), np.dtype[input_x.dtype]]
        vec_ty = np.ndarray[(padded_size,), np.dtype[input_x.dtype]]
        param_ty = np.ndarray[(2 * padded_size,), np.dtype[input_x.dtype]]  # interleaved gamma/beta

        # Object FIFOs - only 3 to stay within DMA limits
        of_x = ObjectFifo(vec_ty, name="x")
        of_y = ObjectFifo(vec_ty, name="y")
        of_params = ObjectFifo(param_ty, name="params")

        # NPU worker: affine transform y = x_norm * gamma + beta
        # x_norm already computed on CPU
        def core_fn(of_x, of_params, of_y):
            # Load params (gamma/beta interleaved) once
            params = of_params.acquire(1)

            for _ in range_(n_vectors):
                vec_x = of_x.acquire(1)
                vec_y = of_y.acquire(1)

                # Affine transform: y = x * gamma + beta
                for i in range_(padded_size):
                    # params[2*i] = gamma, params[2*i+1] = beta
                    vec_y[i] = vec_x[i] * params[2 * i] + params[2 * i + 1]

                of_x.release(1)
                of_y.release(1)

            of_params.release(1)

        worker = Worker(core_fn, fn_args=[of_x.cons(), of_params.cons(), of_y.prod()])

        # Runtime
        rt = Runtime()
        with rt.sequence(tensor_ty, param_ty, tensor_ty) as (x_host, p_host, y_host):
            rt.start(worker)
            rt.fill(of_params.prod(), p_host)
            rt.fill(of_x.prod(), x_host)
            rt.drain(of_y.cons(), y_host, wait=True)

        return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())

    # Convert numpy arrays to iron tensors
    x_iron = iron.zeros((n_elements,), dtype=input_x.dtype)
    params_iron = iron.zeros((2 * padded_size,), dtype=input_x.dtype)
    y_iron = iron.zeros((n_elements,), dtype=input_x.dtype)

    # Copy data into iron tensors and sync to device
    np.copyto(np.asarray(x_iron), x_flat)
    np.copyto(np.asarray(params_iron), params_interleaved)
    x_iron._sync_to_device()
    params_iron._sync_to_device()

    # Execute on NPU
    layernorm_kernel_jit(x_iron, params_iron, y_iron)

    # Sync output from device
    y_iron._sync_from_device()

    # Convert back to numpy and reshape
    y_flat = np.array(y_iron)
    y_padded = y_flat.reshape(n_vectors, padded_size)

    # Unpad and reshape back
    y_2d = y_padded[:, :norm_size]
    y_transposed = y_2d.reshape(orig_shape)

    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    y_result = np.transpose(y_transposed, inv_perm)

    return y_result


def layernorm_cpu(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, axis: int = -1, eps: float = 1e-5) -> np.ndarray:
    """CPU reference implementation."""
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * gamma + beta


def main():
    parser = argparse.ArgumentParser(description="Test layernorm NPU kernel")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--dim", type=int, default=384, help="Normalization dimension")
    parser.add_argument("--scalar", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="f32")
    args = parser.parse_args()

    dtype = np.float16 if args.dtype == "bf16" else np.float32
    np.random.seed(42)
    x = np.random.randn(args.batch, args.dim).astype(dtype)
    gamma = np.ones(args.dim, dtype=dtype)
    beta = np.zeros(args.dim, dtype=dtype)

    print(f"Testing layernorm with shape ({args.batch}, {args.dim}) {args.dtype}")

    if not IRON_AVAILABLE:
        print("IRON not available, running CPU reference only")
        y_cpu = layernorm_cpu(x.astype(np.float32), gamma.astype(np.float32), beta.astype(np.float32)).astype(dtype)
        print(f"CPU result[0]: {y_cpu[0, :5]}...")
        return 0

    try:
        y_npu = layernorm_iron(
            x, gamma, beta,
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
        y_cpu = layernorm_cpu(x.astype(np.float32), gamma.astype(np.float32), beta.astype(np.float32)).astype(dtype)
        max_diff = np.abs(y_npu.astype(np.float32) - y_cpu.astype(np.float32)).max()
        mean_diff = np.abs(y_npu.astype(np.float32) - y_cpu.astype(np.float32)).mean()

        if max_diff < 0.1:
            print(f"PASS: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            return 0
        else:
            print(f"FAIL: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
