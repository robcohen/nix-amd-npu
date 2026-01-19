#!/usr/bin/env python3
"""
NPU integration tests.

Tests that verify NPU kernels work correctly and match CPU results.
Run with: python tests/test_npu.py
"""

import sys
import time
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from kernels import npu_available, matmul_bf16, gelu_bf16, softmax_bf16, layernorm_bf16
from kernels.matmul import MatMulBF16


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_npu_detection():
    """Test NPU detection."""
    print_header("NPU Detection")

    available = npu_available()
    print(f"NPU Available: {available}")

    if available:
        try:
            from kernels.npu_runtime import get_runtime
            runtime = get_runtime()
            print(f"XRT Device: Initialized")
            print(f"Cache Dir: {runtime.cache_dir}")
        except Exception as e:
            print(f"Runtime init error: {e}")
    else:
        print("Running in CPU-only mode (numpy fallback)")

    return available


def test_matmul_correctness():
    """Test matrix multiplication correctness."""
    print_header("MatMul Correctness Test")

    # Test various sizes
    test_cases = [
        (64, 64, 64, "Small square"),
        (128, 64, 128, "Rectangular"),
        (384, 384, 384, "Whisper d_model"),
        (100, 384, 384, "Non-aligned"),
    ]

    all_passed = True
    for M, K, N, name in test_cases:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        # CPU reference
        expected = A @ B

        # Our implementation
        result = matmul_bf16(A, B)

        # Check correctness
        max_error = np.max(np.abs(result - expected))
        rel_error = max_error / (np.max(np.abs(expected)) + 1e-8)

        passed = rel_error < 1e-3  # 0.1% relative error tolerance
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} ({M}x{K} @ {K}x{N}): rel_error={rel_error:.2e}")

        if not passed:
            all_passed = False

    return all_passed


def test_matmul_performance():
    """Benchmark matrix multiplication."""
    print_header("MatMul Performance Benchmark")

    # Whisper-tiny attention dimensions
    sizes = [
        (384, 384, 384, "Linear projection"),
        (1500, 384, 384, "Encoder attention"),
        (448, 384, 384, "Decoder attention"),
    ]

    warmup = 3
    iterations = 10

    for M, K, N, name in sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            _ = matmul_bf16(A, B)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            _ = matmul_bf16(A, B)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / iterations) * 1000
        gflops = (2 * M * K * N) / (elapsed / iterations) / 1e9

        print(f"  {name} ({M}x{K} @ {K}x{N}):")
        print(f"    Time: {avg_ms:.2f} ms/iter")
        print(f"    Perf: {gflops:.2f} GFLOPS")


def test_gelu_correctness():
    """Test GELU activation correctness."""
    print_header("GELU Correctness Test")

    # Reference GELU (exact)
    def gelu_ref(x):
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    sizes = [100, 1000, 10000, 384 * 1500]

    all_passed = True
    for size in sizes:
        x = np.random.randn(size).astype(np.float32)

        expected = gelu_ref(x)
        result = gelu_bf16(x)

        max_error = np.max(np.abs(result - expected))
        rel_error = max_error / (np.max(np.abs(expected)) + 1e-8)

        passed = rel_error < 1e-3
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] size={size}: rel_error={rel_error:.2e}")

        if not passed:
            all_passed = False

    return all_passed


def test_softmax_correctness():
    """Test softmax correctness."""
    print_header("Softmax Correctness Test")

    def softmax_ref(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    test_cases = [
        ((10, 20), "Small 2D"),
        ((6, 100, 100), "Attention scores"),
        ((1, 6, 1500, 1500), "Full encoder attention"),
    ]

    all_passed = True
    for shape, name in test_cases:
        x = np.random.randn(*shape).astype(np.float32)

        expected = softmax_ref(x)
        result = softmax_bf16(x)

        max_error = np.max(np.abs(result - expected))

        # Check sums to 1
        sums = np.sum(result, axis=-1)
        sum_error = np.max(np.abs(sums - 1.0))

        passed = max_error < 1e-4 and sum_error < 1e-5
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} {shape}: max_err={max_error:.2e}, sum_err={sum_error:.2e}")

        if not passed:
            all_passed = False

    return all_passed


def test_layernorm_correctness():
    """Test layer normalization correctness."""
    print_header("LayerNorm Correctness Test")

    test_cases = [
        ((2, 10, 384), 384, "Whisper hidden"),
        ((1, 1500, 384), 384, "Encoder sequence"),
    ]

    all_passed = True
    for shape, norm_dim, name in test_cases:
        x = np.random.randn(*shape).astype(np.float32)
        weight = np.random.randn(norm_dim).astype(np.float32)
        bias = np.random.randn(norm_dim).astype(np.float32)

        result = layernorm_bf16(x, weight, bias)

        # Check output shape
        assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"

        # Check normalized (before affine transform)
        # After layernorm with weight=1, bias=0, mean should be ~0, var should be ~1
        x_normed = layernorm_bf16(x, np.ones(norm_dim), np.zeros(norm_dim))
        mean = np.mean(x_normed, axis=-1)
        var = np.var(x_normed, axis=-1)

        mean_err = np.max(np.abs(mean))
        var_err = np.max(np.abs(var - 1.0))

        passed = mean_err < 1e-4 and var_err < 1e-3
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} {shape}: mean_err={mean_err:.2e}, var_err={var_err:.2e}")

        if not passed:
            all_passed = False

    return all_passed


def test_end_to_end_attention():
    """Test a complete attention computation."""
    print_header("End-to-End Attention Test")

    # Whisper-tiny dimensions
    batch = 1
    n_heads = 6
    seq_len = 100  # Smaller for testing
    head_dim = 64
    d_model = 384

    # Random Q, K, V projections
    Q = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    K = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    V = np.random.randn(batch, seq_len, d_model).astype(np.float32)

    # Reshape for multi-head
    Q = Q.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

    scale = 1.0 / np.sqrt(head_dim)

    # Compute attention step by step
    print("  Computing Q @ K^T...")
    start = time.perf_counter()

    # Manual batched matmul for scores
    scores = np.zeros((batch, n_heads, seq_len, seq_len), dtype=np.float32)
    for b in range(batch):
        for h in range(n_heads):
            scores[b, h] = matmul_bf16(Q[b, h], K[b, h].T.copy())

    scores = scores * scale
    qk_time = time.perf_counter() - start

    print("  Computing softmax...")
    start = time.perf_counter()
    attn_weights = softmax_bf16(scores, axis=-1)
    softmax_time = time.perf_counter() - start

    print("  Computing attention @ V...")
    start = time.perf_counter()
    attn_output = np.zeros((batch, n_heads, seq_len, head_dim), dtype=np.float32)
    for b in range(batch):
        for h in range(n_heads):
            attn_output[b, h] = matmul_bf16(attn_weights[b, h], V[b, h])
    av_time = time.perf_counter() - start

    total_time = qk_time + softmax_time + av_time

    print(f"\n  Results:")
    print(f"    Output shape: {attn_output.shape}")
    print(f"    Q@K^T time: {qk_time*1000:.2f} ms")
    print(f"    Softmax time: {softmax_time*1000:.2f} ms")
    print(f"    Attn@V time: {av_time*1000:.2f} ms")
    print(f"    Total: {total_time*1000:.2f} ms")

    # Verify attention weights sum to 1
    weight_sums = np.sum(attn_weights, axis=-1)
    sum_error = np.max(np.abs(weight_sums - 1.0))
    passed = sum_error < 1e-4

    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] Attention weights sum error: {sum_error:.2e}")

    return passed


def main():
    print("\n" + "="*60)
    print("  WHISPER-IRON NPU TEST SUITE")
    print("="*60)

    results = {}

    # Detection
    npu = test_npu_detection()
    results['NPU Detection'] = npu

    # Correctness tests
    results['MatMul'] = test_matmul_correctness()
    results['GELU'] = test_gelu_correctness()
    results['Softmax'] = test_softmax_correctness()
    results['LayerNorm'] = test_layernorm_correctness()

    # End-to-end
    results['Attention'] = test_end_to_end_attention()

    # Performance (only if NPU available or for baseline)
    test_matmul_performance()

    # Summary
    print_header("Test Summary")
    all_passed = True
    for name, passed in results.items():
        if name == 'NPU Detection':
            status = "Available" if passed else "CPU-only"
        else:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
        print(f"  {name}: {status}")

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
