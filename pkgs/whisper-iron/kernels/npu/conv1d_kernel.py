#!/usr/bin/env python3
"""
1D Convolution NPU kernel using IRON.

Implements Conv1D for Whisper audio encoder.

Usage:
    python conv1d_kernel.py --verify
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
TILE_LEN = 32


def conv1d_iron(
    input_x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride: int = 1,
    padding: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """
    Apply 1D convolution on NPU.

    Args:
        input_x: Input of shape (in_channels, length) or (batch, in_channels, length)
        weight: Weight of shape (out_channels, in_channels, kernel_size)
        bias: Bias of shape (out_channels,)
        stride: Convolution stride
        padding: Padding size
        verbose: Print debug info

    Returns:
        output: Convolution result
    """
    if not IRON_AVAILABLE:
        raise ImportError("IRON API not available")

    # Handle batch dimension
    if input_x.ndim == 2:
        input_x = input_x[np.newaxis, ...]
        squeeze_batch = True
    else:
        squeeze_batch = False

    batch, in_channels, length = input_x.shape
    out_channels, in_ch, kernel_size = weight.shape

    assert in_ch == in_channels, f"Channel mismatch: {in_ch} vs {in_channels}"
    assert kernel_size == 3, "Only kernel_size=3 supported"

    # Output length
    out_length = (length + 2 * padding - kernel_size) // stride + 1

    if verbose:
        print(f"Running conv1d kernel (inline)")
        print(f"Conv1d: ({in_channels}, {length}) -> ({out_channels}, {out_length})")
        print(f"  kernel={kernel_size}, stride={stride}, padding={padding}")

    # Pad input
    x_padded = np.pad(input_x, ((0, 0), (0, 0), (padding, padding)), mode='constant')

    # Output buffer
    output = np.zeros((batch, out_channels, out_length), dtype=input_x.dtype)

    # For simple implementation, process per batch item
    for b in range(batch):
        x_batch = x_padded[b]  # (in_channels, length + 2*padding)
        padded_len = x_batch.shape[1]

        # Flatten for IRON
        n_x_elements = in_channels * padded_len
        n_w_elements = out_channels * in_channels * kernel_size
        n_b_elements = out_channels
        n_y_elements = out_channels * out_length

        # Get device target (NPU2 is the default)
        device = iron.get_current_device()

        # Combine weight and bias: append bias at end of weight buffer
        n_params = n_w_elements + n_b_elements
        params = np.zeros(n_params, dtype=input_x.dtype)
        params[:n_w_elements] = weight.ravel()
        params[n_w_elements:] = bias.ravel()

        @iron.jit(is_placed=False)
        def conv1d_kernel_jit(x_tensor, params_tensor, y_tensor):
            x_ty = np.ndarray[(n_x_elements,), np.dtype[input_x.dtype]]
            params_ty = np.ndarray[(n_params,), np.dtype[input_x.dtype]]
            y_ty = np.ndarray[(n_y_elements,), np.dtype[input_x.dtype]]

            # Use small tiles for processing
            tile_size = min(TILE_LEN, out_length)
            n_tiles = (out_length + tile_size - 1) // tile_size

            out_tile_ty = np.ndarray[(tile_size * out_channels,), np.dtype[input_x.dtype]]

            # Object FIFOs - only 3 to stay within DMA limits
            of_x = ObjectFifo(x_ty, name="x")
            of_params = ObjectFifo(params_ty, name="params")
            of_y = ObjectFifo(out_tile_ty, name="y")

            # Worker with inline conv1d using array-element accumulation
            def core_fn(of_x, of_params, of_y):
                # Load all inputs once
                x = of_x.acquire(1)
                p = of_params.acquire(1)
                # p[:n_w_elements] = weights, p[n_w_elements:] = bias

                # Process output in tiles
                for t in range_(n_tiles):
                    y_tile = of_y.acquire(1)

                    # Compute convolution for this tile
                    for local_o in range_(tile_size):
                        for oc in range_(out_channels):
                            y_idx = local_o * out_channels + oc
                            # Initialize with bias from params
                            y_tile[y_idx] = p[n_w_elements + oc]
                            # Accumulate convolution
                            for ic in range_(in_channels):
                                for k in range_(kernel_size):
                                    o = t * tile_size + local_o
                                    x_idx = ic * padded_len + o * stride + k
                                    w_idx = oc * in_channels * kernel_size + ic * kernel_size + k
                                    y_tile[y_idx] = y_tile[y_idx] + x[x_idx] * p[w_idx]

                    of_y.release(1)

                of_x.release(1)
                of_params.release(1)

            worker = Worker(core_fn, fn_args=[of_x.cons(), of_params.cons(), of_y.prod()])

            # Runtime
            rt = Runtime()
            with rt.sequence(x_ty, params_ty, y_ty) as (x_h, p_h, y_h):
                rt.start(worker)
                rt.fill(of_x.prod(), x_h)
                rt.fill(of_params.prod(), p_h)
                rt.drain(of_y.cons(), y_h, wait=True)

            return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())

        # Convert numpy arrays to iron tensors
        x_iron = iron.zeros((n_x_elements,), dtype=input_x.dtype)
        params_iron = iron.zeros((n_params,), dtype=input_x.dtype)
        y_iron = iron.zeros((n_y_elements,), dtype=input_x.dtype)

        # Copy data into iron tensors and sync to device
        np.copyto(np.asarray(x_iron), x_batch.ravel())
        np.copyto(np.asarray(params_iron), params)
        x_iron._sync_to_device()
        params_iron._sync_to_device()

        # Execute on NPU
        conv1d_kernel_jit(x_iron, params_iron, y_iron)

        # Sync output from device
        y_iron._sync_from_device()

        # Convert back to numpy and reshape
        output[b] = np.array(y_iron).reshape(out_length, out_channels).T

    if squeeze_batch:
        output = output[0]

    return output


def conv1d_cpu(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int = 1, padding: int = 1) -> np.ndarray:
    """CPU reference implementation."""
    if x.ndim == 2:
        x = x[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    batch, in_ch, length = x.shape
    out_ch, _, kernel = weight.shape

    # Pad input
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')

    # Output length
    out_len = (length + 2 * padding - kernel) // stride + 1
    output = np.zeros((batch, out_ch, out_len), dtype=x.dtype)

    # Naive convolution
    for b in range(batch):
        for oc in range(out_ch):
            for o in range(out_len):
                acc = float(bias[oc])
                for ic in range(in_ch):
                    for k in range(kernel):
                        acc += float(x_padded[b, ic, o * stride + k]) * float(weight[oc, ic, k])
                output[b, oc, o] = acc

    if squeeze:
        output = output[0]

    return output


def main():
    parser = argparse.ArgumentParser(description="Test conv1d NPU kernel")
    parser.add_argument("--in-ch", type=int, default=8, help="Input channels")
    parser.add_argument("--out-ch", type=int, default=16, help="Output channels")
    parser.add_argument("--length", type=int, default=64, help="Input length")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="f32")
    args = parser.parse_args()

    dtype = np.float16 if args.dtype == "bf16" else np.float32
    np.random.seed(42)

    x = np.random.randn(args.in_ch, args.length).astype(dtype) * 0.1
    weight = np.random.randn(args.out_ch, args.in_ch, 3).astype(dtype) * 0.1
    bias = np.random.randn(args.out_ch).astype(dtype) * 0.01

    print(f"Testing conv1d: ({args.in_ch}, {args.length}) -> ({args.out_ch}, ?)")

    if not IRON_AVAILABLE:
        print("IRON not available, running CPU reference only")
        y_cpu = conv1d_cpu(x.astype(np.float32), weight.astype(np.float32), bias.astype(np.float32), stride=args.stride).astype(dtype)
        print(f"CPU output shape: {y_cpu.shape}")
        print(f"CPU result[0, :5]: {y_cpu[0, :5]}")
        return 0

    try:
        import time
        start = time.perf_counter()
        y_npu = conv1d_iron(x, weight, bias, stride=args.stride, verbose=args.verbose)
        elapsed = time.perf_counter() - start
        print(f"NPU output shape: {y_npu.shape}")
        print(f"NPU result[0, :5]: {y_npu[0, :5]}")
        print(f"NPU time: {elapsed*1000:.1f}ms")
    except Exception as e:
        print(f"NPU execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    if args.verify:
        y_cpu = conv1d_cpu(x.astype(np.float32), weight.astype(np.float32), bias.astype(np.float32), stride=args.stride).astype(dtype)
        max_diff = np.abs(y_npu.astype(np.float32) - y_cpu.astype(np.float32)).max()
        mean_diff = np.abs(y_npu.astype(np.float32) - y_cpu.astype(np.float32)).mean()

        if max_diff < 0.1:
            print(f"PASS: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}")
            return 0
        else:
            print(f"FAIL: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
