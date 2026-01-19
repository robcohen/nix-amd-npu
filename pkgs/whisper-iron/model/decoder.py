"""
Whisper text decoder.
"""

import numpy as np
from typing import List, Optional

from .config import WhisperConfig
from .attention import MultiHeadAttention, CrossAttention
from .mlp import MLP
from kernels import add_bf16, linear_bf16
from kernels.layernorm import LayerNormBF16

BF16_DTYPE = np.float32


class DecoderLayer:
    """
    Single decoder transformer layer.

    Structure:
        x -> LayerNorm -> Causal Self-Attention -> + (residual)
        x -> LayerNorm -> Cross-Attention (to encoder) -> + (residual)
        x -> LayerNorm -> MLP -> + (residual)
    """

    def __init__(self, config: WhisperConfig):
        self.config = config

        # Layer norms
        self.self_attn_ln = LayerNormBF16(config.d_model)
        self.cross_attn_ln = LayerNormBF16(config.d_model)
        self.mlp_ln = LayerNormBF16(config.d_model)

        # Causal self-attention
        self.self_attn = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            is_causal=True,
        )

        # Cross-attention to encoder
        self.cross_attn = CrossAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
        )

        # MLP
        self.mlp = MLP(config.d_model, config.d_ff)

    def __call__(
        self,
        x: np.ndarray,
        encoder_output: np.ndarray,
        encoder_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass through decoder layer.

        Args:
            x: Input of shape (batch, tgt_len, d_model)
            encoder_output: Encoder output of shape (batch, src_len, d_model)
            encoder_mask: Optional mask for cross-attention

        Returns:
            Output of shape (batch, tgt_len, d_model)
        """
        # Causal self-attention with residual
        residual = x
        x = self.self_attn_ln(x)
        x = self.self_attn(x)
        x = add_bf16(residual, x)

        # Cross-attention with residual
        residual = x
        x = self.cross_attn_ln(x)
        x = self.cross_attn(x, encoder_output, mask=encoder_mask)
        x = add_bf16(residual, x)

        # MLP with residual
        residual = x
        x = self.mlp_ln(x)
        x = self.mlp(x)
        x = add_bf16(residual, x)

        return x


class WhisperDecoder:
    """
    Whisper text decoder.

    Structure:
        - Token embedding
        - Positional embedding
        - N x DecoderLayer
        - LayerNorm
        - Output projection (to vocab)
    """

    def __init__(self, config: WhisperConfig):
        self.config = config

        # Token embedding
        self.embed_tokens = np.zeros((config.vocab_size, config.d_model), dtype=BF16_DTYPE)

        # Positional embedding
        self.embed_positions = np.zeros((config.max_target_positions, config.d_model), dtype=BF16_DTYPE)

        # Transformer layers
        self.layers: List[DecoderLayer] = [
            DecoderLayer(config) for _ in range(config.decoder_layers)
        ]

        # Final layer norm
        self.ln = LayerNormBF16(config.d_model)

        # Output projection (tied with token embedding in original Whisper)
        # proj_out: (d_model) -> (vocab_size)
        self.use_tied_weights = True

    def __call__(
        self,
        input_ids: np.ndarray,
        encoder_output: np.ndarray,
        encoder_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Decode tokens given encoder output.

        Args:
            input_ids: Token IDs of shape (batch, tgt_len)
            encoder_output: Encoder output of shape (batch, src_len, d_model)
            encoder_mask: Optional mask for cross-attention

        Returns:
            Logits of shape (batch, tgt_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Token embedding lookup
        x = self.embed_tokens[input_ids]  # (batch, seq_len, d_model)

        # Add positional embedding
        positions = np.arange(seq_len)
        x = add_bf16(x, self.embed_positions[positions])

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask)

        # Final layer norm
        x = self.ln(x)

        # Output projection to vocabulary
        # x: (batch, seq_len, d_model)
        # embed_tokens: (vocab_size, d_model)
        # logits: (batch, seq_len, vocab_size) = x @ embed_tokens.T
        if self.use_tied_weights:
            logits = linear_bf16(x, self.embed_tokens)
        else:
            # If not tied, would need separate proj_out weight
            logits = linear_bf16(x, self.embed_tokens)

        return logits

    def load_weights(self, state_dict: dict):
        """
        Load weights from HuggingFace state dict.

        Args:
            state_dict: Dictionary of weight tensors
        """
        # Embeddings
        self.embed_tokens = state_dict["embed_tokens.weight"].astype(BF16_DTYPE)
        self.embed_positions = state_dict["embed_positions.weight"].astype(BF16_DTYPE)

        # Layer weights
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."

            # Layer norms
            layer.self_attn_ln.load_weights(
                state_dict[prefix + "self_attn_layer_norm.weight"],
                state_dict[prefix + "self_attn_layer_norm.bias"],
            )
            layer.cross_attn_ln.load_weights(
                state_dict[prefix + "encoder_attn_layer_norm.weight"],
                state_dict[prefix + "encoder_attn_layer_norm.bias"],
            )
            layer.mlp_ln.load_weights(
                state_dict[prefix + "final_layer_norm.weight"],
                state_dict[prefix + "final_layer_norm.bias"],
            )

            # Self-attention (has Q and V biases, no K bias)
            layer.self_attn.load_weights(
                q_proj_weight=state_dict[prefix + "self_attn.q_proj.weight"],
                k_proj_weight=state_dict[prefix + "self_attn.k_proj.weight"],
                v_proj_weight=state_dict[prefix + "self_attn.v_proj.weight"],
                out_proj_weight=state_dict[prefix + "self_attn.out_proj.weight"],
                out_proj_bias=state_dict[prefix + "self_attn.out_proj.bias"],
                q_proj_bias=state_dict.get(prefix + "self_attn.q_proj.bias"),
                v_proj_bias=state_dict.get(prefix + "self_attn.v_proj.bias"),
            )

            # Cross-attention (has Q and V biases, no K bias)
            layer.cross_attn.load_weights(
                q_proj_weight=state_dict[prefix + "encoder_attn.q_proj.weight"],
                k_proj_weight=state_dict[prefix + "encoder_attn.k_proj.weight"],
                v_proj_weight=state_dict[prefix + "encoder_attn.v_proj.weight"],
                out_proj_weight=state_dict[prefix + "encoder_attn.out_proj.weight"],
                out_proj_bias=state_dict[prefix + "encoder_attn.out_proj.bias"],
                q_proj_bias=state_dict.get(prefix + "encoder_attn.q_proj.bias"),
                v_proj_bias=state_dict.get(prefix + "encoder_attn.v_proj.bias"),
            )

            # MLP
            layer.mlp.load_weights(
                fc1_weight=state_dict[prefix + "fc1.weight"],
                fc1_bias=state_dict[prefix + "fc1.bias"],
                fc2_weight=state_dict[prefix + "fc2.weight"],
                fc2_bias=state_dict[prefix + "fc2.bias"],
            )

        # Final layer norm
        self.ln.load_weights(
            state_dict["layer_norm.weight"],
            state_dict["layer_norm.bias"],
        )
