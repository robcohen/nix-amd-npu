"""
Multi-head attention for Whisper.
"""

import numpy as np
from typing import Optional, Tuple

from kernels import linear_bf16, softmax_bf16, matmul_bf16, add_bf16
from kernels.softmax import masked_softmax_bf16, causal_mask

BF16_DTYPE = np.float32


class MultiHeadAttention:
    """
    Multi-head attention mechanism.

    Supports both self-attention and cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        is_causal: bool = False,
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            is_causal: Whether to apply causal masking (for decoder self-attention)
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.is_causal = is_causal
        self.scale = 1.0 / np.sqrt(self.head_dim)

        # Projection weights (will be loaded from checkpoint)
        # Q, K, V projections (with optional biases)
        self.q_proj_weight = np.zeros((d_model, d_model), dtype=BF16_DTYPE)
        self.q_proj_bias = None  # Optional
        self.k_proj_weight = np.zeros((d_model, d_model), dtype=BF16_DTYPE)
        self.k_proj_bias = None  # Optional (usually None for cross-attention K)
        self.v_proj_weight = np.zeros((d_model, d_model), dtype=BF16_DTYPE)
        self.v_proj_bias = None  # Optional

        # Output projection
        self.out_proj_weight = np.zeros((d_model, d_model), dtype=BF16_DTYPE)
        self.out_proj_bias = np.zeros(d_model, dtype=BF16_DTYPE)

    def load_weights(
        self,
        q_proj_weight: np.ndarray,
        k_proj_weight: np.ndarray,
        v_proj_weight: np.ndarray,
        out_proj_weight: np.ndarray,
        out_proj_bias: np.ndarray,
        q_proj_bias: Optional[np.ndarray] = None,
        k_proj_bias: Optional[np.ndarray] = None,
        v_proj_bias: Optional[np.ndarray] = None,
    ):
        """Load weights from checkpoint."""
        self.q_proj_weight = q_proj_weight.astype(BF16_DTYPE)
        self.k_proj_weight = k_proj_weight.astype(BF16_DTYPE)
        self.v_proj_weight = v_proj_weight.astype(BF16_DTYPE)
        self.out_proj_weight = out_proj_weight.astype(BF16_DTYPE)
        self.out_proj_bias = out_proj_bias.astype(BF16_DTYPE)

        # Optional biases
        self.q_proj_bias = q_proj_bias.astype(BF16_DTYPE) if q_proj_bias is not None else None
        self.k_proj_bias = k_proj_bias.astype(BF16_DTYPE) if k_proj_bias is not None else None
        self.v_proj_bias = v_proj_bias.astype(BF16_DTYPE) if v_proj_bias is not None else None

    def __call__(
        self,
        query: np.ndarray,
        key: Optional[np.ndarray] = None,
        value: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply multi-head attention.

        For self-attention: key=value=None, uses query for all
        For cross-attention: key and value come from encoder

        Args:
            query: Query tensor of shape (batch, seq_len, d_model)
            key: Key tensor (for cross-attention), shape (batch, kv_len, d_model)
            value: Value tensor (for cross-attention), shape (batch, kv_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Self-attention: use query for key and value
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, seq_len, _ = query.shape
        _, kv_len, _ = key.shape

        # Project Q, K, V (with optional biases)
        q = linear_bf16(query, self.q_proj_weight, self.q_proj_bias)  # (batch, seq_len, d_model)
        k = linear_bf16(key, self.k_proj_weight, self.k_proj_bias)    # (batch, kv_len, d_model)
        v = linear_bf16(value, self.v_proj_weight, self.v_proj_bias)  # (batch, kv_len, d_model)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)

        k = k.reshape(batch_size, kv_len, self.n_heads, self.head_dim)
        k = k.transpose(0, 2, 1, 3)  # (batch, n_heads, kv_len, head_dim)

        v = v.reshape(batch_size, kv_len, self.n_heads, self.head_dim)
        v = v.transpose(0, 2, 1, 3)  # (batch, n_heads, kv_len, head_dim)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # (batch, n_heads, seq_len, head_dim) @ (batch, n_heads, head_dim, kv_len)
        # -> (batch, n_heads, seq_len, kv_len)
        scores = np.einsum('bhqd,bhkd->bhqk', q, k) * self.scale

        # Apply causal mask if needed
        if self.is_causal:
            causal = causal_mask(seq_len)
            # Expand mask for batch and heads
            causal = causal.reshape(1, 1, seq_len, seq_len)
            scores = np.where(causal, -1e9, scores)

        # Apply additional mask if provided
        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        # Softmax over key dimension
        attn_weights = softmax_bf16(scores, axis=-1)

        # Apply attention to values
        # (batch, n_heads, seq_len, kv_len) @ (batch, n_heads, kv_len, head_dim)
        # -> (batch, n_heads, seq_len, head_dim)
        attn_output = np.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        # Reshape back: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3)  # (batch, seq_len, n_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        # Output projection
        output = linear_bf16(attn_output, self.out_proj_weight, self.out_proj_bias)

        return output


class CrossAttention(MultiHeadAttention):
    """
    Cross-attention for decoder attending to encoder output.

    Same as MultiHeadAttention but always uses separate key/value from encoder.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__(d_model, n_heads, is_causal=False)

    def __call__(
        self,
        query: np.ndarray,
        encoder_output: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply cross-attention.

        Args:
            query: Decoder query of shape (batch, tgt_len, d_model)
            encoder_output: Encoder output of shape (batch, src_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, tgt_len, d_model)
        """
        return super().__call__(
            query=query,
            key=encoder_output,
            value=encoder_output,
            mask=mask,
        )
