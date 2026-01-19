#!/usr/bin/env python3
"""
Benchmark suite for Whisper-IRON NPU kernels.

Compares CPU vs NPU performance for all kernel types.

Usage:
    python benchmark.py [--all] [--kernel NAME] [--size SIZE]
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def benchmark_kernel(name: str, cpu_fn, npu_fn, setup_fn, sizes: list, n_warmup: int = 2, n_iter: int = 10):
    """Benchmark a single kernel across different sizes."""
    results = []

    for size in sizes:
        # Setup
        args = setup_fn(size)

        # Warmup CPU
        for _ in range(n_warmup):
            cpu_fn(*args)

        # Benchmark CPU
        cpu_times = []
        for _ in range(n_iter):
            start = time.perf_counter()
            cpu_fn(*args)
            cpu_times.append(time.perf_counter() - start)
        cpu_avg = np.mean(cpu_times) * 1000  # ms

        # Try NPU
        npu_avg = None
        speedup = None
        try:
            # Warmup NPU
            for _ in range(n_warmup):
                npu_fn(*args)

            # Benchmark NPU
            npu_times = []
            for _ in range(n_iter):
                start = time.perf_counter()
                npu_fn(*args)
                npu_times.append(time.perf_counter() - start)
            npu_avg = np.mean(npu_times) * 1000  # ms
            speedup = cpu_avg / npu_avg
        except Exception as e:
            npu_avg = f"Error: {e}"

        results.append({
            "size": size,
            "cpu_ms": cpu_avg,
            "npu_ms": npu_avg,
            "speedup": speedup,
        })

    return results


def print_results(name: str, results: list):
    """Print benchmark results in a table."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"{'Size':<20} {'CPU (ms)':<12} {'NPU (ms)':<12} {'Speedup':<10}")
    print("-" * 60)

    for r in results:
        size_str = str(r['size'])
        cpu_str = f"{r['cpu_ms']:.3f}"
        if isinstance(r['npu_ms'], float):
            npu_str = f"{r['npu_ms']:.3f}"
            speedup_str = f"{r['speedup']:.2f}x"
        else:
            npu_str = "N/A"
            speedup_str = "-"
        print(f"{size_str:<20} {cpu_str:<12} {npu_str:<12} {speedup_str:<10}")


def benchmark_add():
    """Benchmark vector addition."""
    from kernels import add_bf16

    def setup(size):
        a = np.random.randn(size).astype(np.float32)
        b = np.random.randn(size).astype(np.float32)
        return (a, b)

    def cpu_fn(a, b):
        return add_bf16(a, b, use_npu=False)

    def npu_fn(a, b):
        return add_bf16(a, b, use_npu=True)

    sizes = [1024, 4096, 16384, 65536, 262144, 1048576]
    return benchmark_kernel("add_bf16", cpu_fn, npu_fn, setup, sizes)


def benchmark_gelu():
    """Benchmark GELU activation."""
    from kernels import gelu_bf16

    def setup(size):
        x = np.random.randn(size).astype(np.float32)
        return (x,)

    def cpu_fn(x):
        return gelu_bf16(x, use_npu=False)

    def npu_fn(x):
        return gelu_bf16(x, use_npu=True)

    sizes = [1024, 4096, 16384, 65536, 262144]
    return benchmark_kernel("gelu_bf16", cpu_fn, npu_fn, setup, sizes)


def benchmark_softmax():
    """Benchmark softmax."""
    from kernels import softmax_bf16

    def setup(size):
        rows, cols = size
        x = np.random.randn(rows, cols).astype(np.float32)
        return (x,)

    def cpu_fn(x):
        return softmax_bf16(x, use_npu=False)

    def npu_fn(x):
        return softmax_bf16(x, use_npu=True)

    sizes = [(16, 64), (64, 128), (256, 256), (512, 512), (1500, 1500)]
    return benchmark_kernel("softmax_bf16", cpu_fn, npu_fn, setup, sizes)


def benchmark_layernorm():
    """Benchmark layer normalization."""
    from kernels.layernorm import layernorm_bf16

    def setup(size):
        batch, dim = size
        x = np.random.randn(batch, dim).astype(np.float32)
        weight = np.ones(dim, dtype=np.float32)
        bias = np.zeros(dim, dtype=np.float32)
        return (x, weight, bias)

    def cpu_fn(x, w, b):
        return layernorm_bf16(x, w, b, use_npu=False)

    def npu_fn(x, w, b):
        return layernorm_bf16(x, w, b, use_npu=True)

    sizes = [(64, 384), (256, 384), (1024, 384), (1500, 384), (1500, 768)]
    return benchmark_kernel("layernorm_bf16", cpu_fn, npu_fn, setup, sizes)


def benchmark_matmul():
    """Benchmark matrix multiplication."""
    from kernels import matmul_bf16

    def setup(size):
        m, k, n = size
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        return (a, b)

    def cpu_fn(a, b):
        return matmul_bf16(a, b, use_npu=False)

    def npu_fn(a, b):
        return matmul_bf16(a, b, use_npu=True)

    sizes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (384, 384, 384),      # Whisper attention
        (1500, 384, 384),     # Whisper Q/K/V projection
        (1500, 384, 1536),    # Whisper MLP fc1
    ]
    return benchmark_kernel("matmul_bf16", cpu_fn, npu_fn, setup, sizes)


def benchmark_conv1d():
    """Benchmark 1D convolution."""
    from kernels import conv1d_bf16

    def setup(size):
        in_ch, out_ch, length = size
        x = np.random.randn(1, in_ch, length).astype(np.float32)
        weight = np.random.randn(out_ch, in_ch, 3).astype(np.float32)
        bias = np.random.randn(out_ch).astype(np.float32)
        return (x, weight, bias)

    def cpu_fn(x, w, b):
        return conv1d_bf16(x, w, b, stride=1, padding=1, use_npu=False)

    def npu_fn(x, w, b):
        return conv1d_bf16(x, w, b, stride=1, padding=1, use_npu=True)

    sizes = [
        (80, 384, 3000),   # Whisper layer 1
        (384, 384, 3000),  # Whisper layer 2
        (80, 384, 1500),
        (384, 384, 1500),
    ]
    return benchmark_kernel("conv1d_bf16", cpu_fn, npu_fn, setup, sizes)


def benchmark_whisper_encoder():
    """Benchmark full Whisper encoder pass."""
    try:
        from model import Whisper
        from utils import preprocess_audio

        print("\n" + "=" * 60)
        print("  Whisper Encoder Benchmark")
        print("=" * 60)

        # Create dummy mel spectrogram
        mel = np.random.randn(1, 80, 3000).astype(np.float32)

        # Load model
        print("Loading whisper-tiny model...")
        model = Whisper.from_pretrained("openai/whisper-tiny")

        # Warmup
        for _ in range(2):
            _ = model.encode(mel)

        # Benchmark
        n_iter = 5
        times = []
        for _ in range(n_iter):
            start = time.perf_counter()
            _ = model.encode(mel)
            times.append(time.perf_counter() - start)

        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000

        print(f"Encoder latency: {avg_ms:.1f} ± {std_ms:.1f} ms")
        print(f"Throughput: {1000/avg_ms:.1f} inferences/sec")

    except Exception as e:
        print(f"Whisper benchmark failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisper-IRON kernels")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--kernel", type=str, help="Specific kernel to benchmark")
    parser.add_argument("--whisper", action="store_true", help="Benchmark Whisper encoder")
    args = parser.parse_args()

    print("=" * 60)
    print("  WHISPER-IRON KERNEL BENCHMARKS")
    print("=" * 60)

    # Check NPU status
    try:
        from kernels import get_npu_status
        status = get_npu_status()
        print(f"\nNPU Status:")
        print(f"  IRON available: {status['iron_available']}")
        print(f"  NPU device: {status['npu_available']}")
        print(f"  Backend: {status['backend']}")
    except Exception as e:
        print(f"\nCould not get NPU status: {e}")

    benchmarks = {
        "add": benchmark_add,
        "gelu": benchmark_gelu,
        "softmax": benchmark_softmax,
        "layernorm": benchmark_layernorm,
        "matmul": benchmark_matmul,
        "conv1d": benchmark_conv1d,
    }

    if args.kernel:
        if args.kernel in benchmarks:
            results = benchmarks[args.kernel]()
            print_results(args.kernel, results)
        else:
            print(f"Unknown kernel: {args.kernel}")
            print(f"Available: {list(benchmarks.keys())}")
            return 1
    elif args.whisper:
        benchmark_whisper_encoder()
    elif args.all:
        for name, fn in benchmarks.items():
            try:
                results = fn()
                print_results(name, results)
            except Exception as e:
                print(f"\n{name}: FAILED - {e}")
        benchmark_whisper_encoder()
    else:
        # Default: run matmul and show summary
        print("\nRunning matmul benchmark (use --all for full suite)...")
        results = benchmark_matmul()
        print_results("matmul_bf16", results)

    print("\n" + "=" * 60)
    print("  Benchmark complete")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
