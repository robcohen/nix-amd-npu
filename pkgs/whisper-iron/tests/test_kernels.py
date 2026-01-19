"""
Tests for NPU kernels.

Run with: pytest tests/test_kernels.py -v
"""

import numpy as np
import pytest

# Import kernels
from kernels import matmul_bf16, linear_bf16, add_bf16, gelu_bf16, layernorm_bf16, softmax_bf16
from kernels.conv1d import conv1d_bf16, conv1d_im2col_bf16


class TestMatMul:
    """Tests for matrix multiplication kernel."""

    def test_square_matrices(self):
        """Test matmul with square matrices."""
        M, K, N = 64, 64, 64
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        result = matmul_bf16(A, B)
        expected = A @ B

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_rectangular_matrices(self):
        """Test matmul with rectangular matrices."""
        M, K, N = 32, 64, 128
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        result = matmul_bf16(A, B)
        expected = A @ B

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_whisper_dimensions(self):
        """Test with Whisper-like dimensions."""
        # Attention: seq_len x d_model @ d_model x d_model
        seq_len, d_model = 1500, 384
        A = np.random.randn(seq_len, d_model).astype(np.float32)
        B = np.random.randn(d_model, d_model).astype(np.float32)

        result = matmul_bf16(A, B)
        expected = A @ B

        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


class TestLinear:
    """Tests for linear layer."""

    def test_linear_no_bias(self):
        """Test linear without bias."""
        batch, seq, in_f, out_f = 1, 100, 384, 384
        x = np.random.randn(batch, seq, in_f).astype(np.float32)
        weight = np.random.randn(out_f, in_f).astype(np.float32)

        result = linear_bf16(x, weight)
        expected = x @ weight.T

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_linear_with_bias(self):
        """Test linear with bias."""
        batch, seq, in_f, out_f = 1, 100, 384, 384
        x = np.random.randn(batch, seq, in_f).astype(np.float32)
        weight = np.random.randn(out_f, in_f).astype(np.float32)
        bias = np.random.randn(out_f).astype(np.float32)

        result = linear_bf16(x, weight, bias)
        expected = x @ weight.T + bias

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


class TestElementwise:
    """Tests for elementwise operations."""

    def test_add(self):
        """Test elementwise addition."""
        a = np.random.randn(10, 20).astype(np.float32)
        b = np.random.randn(10, 20).astype(np.float32)

        result = add_bf16(a, b)
        expected = a + b

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_add_broadcast(self):
        """Test addition with broadcasting."""
        a = np.random.randn(10, 20).astype(np.float32)
        b = np.random.randn(20).astype(np.float32)

        result = add_bf16(a, b)
        expected = a + b

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_gelu(self):
        """Test GELU activation."""
        x = np.random.randn(100).astype(np.float32)

        result = gelu_bf16(x)

        # GELU should be roughly x for large positive, 0 for large negative
        assert np.all(result[x > 3] > 0)
        assert np.all(np.abs(result[x < -3]) < 0.1)


class TestLayerNorm:
    """Tests for layer normalization."""

    def test_layernorm_basic(self):
        """Test basic layer normalization."""
        x = np.random.randn(2, 10, 384).astype(np.float32)
        weight = np.ones(384, dtype=np.float32)
        bias = np.zeros(384, dtype=np.float32)

        result = layernorm_bf16(x, weight, bias)

        # Check that output is normalized (mean ~0, var ~1)
        mean = np.mean(result, axis=-1)
        var = np.var(result, axis=-1)

        np.testing.assert_allclose(mean, 0, atol=1e-5)
        np.testing.assert_allclose(var, 1, atol=1e-4)

    def test_layernorm_with_affine(self):
        """Test layer norm with non-trivial weight/bias."""
        x = np.random.randn(2, 10, 64).astype(np.float32)
        weight = np.random.randn(64).astype(np.float32)
        bias = np.random.randn(64).astype(np.float32)

        result = layernorm_bf16(x, weight, bias)

        # Should have same shape
        assert result.shape == x.shape


class TestSoftmax:
    """Tests for softmax."""

    def test_softmax_sums_to_one(self):
        """Test that softmax sums to 1."""
        x = np.random.randn(10, 20).astype(np.float32)

        result = softmax_bf16(x, axis=-1)

        # Each row should sum to 1
        sums = np.sum(result, axis=-1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    def test_softmax_non_negative(self):
        """Test that softmax is non-negative."""
        x = np.random.randn(10, 20).astype(np.float32)

        result = softmax_bf16(x, axis=-1)

        assert np.all(result >= 0)

    def test_softmax_numerical_stability(self):
        """Test softmax with large values."""
        x = np.array([[1000, 1001, 1002]], dtype=np.float32)

        result = softmax_bf16(x, axis=-1)

        # Should not overflow/underflow
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_allclose(np.sum(result), 1.0, rtol=1e-5)


class TestConv1d:
    """Tests for 1D convolution."""

    def test_conv1d_basic(self):
        """Test basic conv1d."""
        batch, in_ch, length = 1, 3, 10
        out_ch, kernel = 5, 3

        x = np.random.randn(batch, in_ch, length).astype(np.float32)
        weight = np.random.randn(out_ch, in_ch, kernel).astype(np.float32)

        result = conv1d_bf16(x, weight, stride=1, padding=0)

        # Output length = (10 - 3) // 1 + 1 = 8
        assert result.shape == (batch, out_ch, 8)

    def test_conv1d_with_padding(self):
        """Test conv1d with padding."""
        batch, in_ch, length = 1, 3, 10
        out_ch, kernel = 5, 3

        x = np.random.randn(batch, in_ch, length).astype(np.float32)
        weight = np.random.randn(out_ch, in_ch, kernel).astype(np.float32)

        result = conv1d_bf16(x, weight, stride=1, padding=1)

        # Output length = (10 + 2 - 3) // 1 + 1 = 10
        assert result.shape == (batch, out_ch, 10)

    def test_conv1d_im2col_matches_naive(self):
        """Test that im2col implementation matches naive."""
        batch, in_ch, length = 2, 4, 20
        out_ch, kernel = 8, 3

        x = np.random.randn(batch, in_ch, length).astype(np.float32)
        weight = np.random.randn(out_ch, in_ch, kernel).astype(np.float32)
        bias = np.random.randn(out_ch).astype(np.float32)

        result_naive = conv1d_bf16(x, weight, bias, stride=1, padding=1)
        result_im2col = conv1d_im2col_bf16(x, weight, bias, stride=1, padding=1)

        np.testing.assert_allclose(result_naive, result_im2col, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
