"""
Whisper audio encoder.
"""

import numpy as np
from typing import List

from .config import WhisperConfig
from .attention import MultiHeadAttention
from .mlp import MLP
from kernels import add_bf16, conv1d_bf16
from kernels.layernorm import LayerNormBF16
from kernels.conv1d import Conv1dBF16

BF16_DTYPE = np.float32


class EncoderLayer:
    """
    Single encoder transformer layer.

    Structure:
        x -> LayerNorm -> Self-Attention -> + (residual)
        x -> LayerNorm -> MLP -> + (residual)
    """

    def __init__(self, config: WhisperConfig):
        self.config = config

        # Layer norms
        self.attn_ln = LayerNormBF16(config.d_model)
        self.mlp_ln = LayerNormBF16(config.d_model)

        # Self-attention
        self.self_attn = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            is_causal=False,
        )

        # MLP
        self.mlp = MLP(config.d_model, config.d_ff)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through encoder layer.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # Self-attention block with residual
        residual = x
        x = self.attn_ln(x)
        x = self.self_attn(x)
        x = add_bf16(residual, x)

        # MLP block with residual
        residual = x
        x = self.mlp_ln(x)
        x = self.mlp(x)
        x = add_bf16(residual, x)

        return x


class WhisperEncoder:
    """
    Whisper audio encoder.

    Structure:
        - Conv1d (n_mels -> d_model, kernel=3, padding=1)
        - GELU
        - Conv1d (d_model -> d_model, kernel=3, stride=2, padding=1)
        - GELU
        - Positional embedding
        - N x EncoderLayer
        - LayerNorm
    """

    def __init__(self, config: WhisperConfig):
        self.config = config

        # Convolutional frontend
        self.conv1 = Conv1dBF16(
            in_channels=config.n_mels,
            out_channels=config.d_model,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = Conv1dBF16(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Positional embedding
        # Max position = max_source_positions // 2 (due to stride=2 in conv2)
        max_pos = config.max_source_positions // 2
        self.positional_embedding = np.zeros((max_pos, config.d_model), dtype=BF16_DTYPE)

        # Transformer layers
        self.layers: List[EncoderLayer] = [
            EncoderLayer(config) for _ in range(config.encoder_layers)
        ]

        # Final layer norm
        self.ln_post = LayerNormBF16(config.d_model)

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """
        Encode audio mel spectrogram.

        Args:
            mel: Mel spectrogram of shape (batch, n_mels, n_frames)
                 where n_frames <= max_source_positions

        Returns:
            Encoder output of shape (batch, seq_len, d_model)
            where seq_len = n_frames // 2 (due to stride-2 conv)
        """
        from kernels.elementwise import gelu_bf16

        batch_size = mel.shape[0]

        # Conv1d expects (batch, channels, length)
        x = mel.astype(BF16_DTYPE)

        # First conv + GELU
        x = self.conv1(x)
        x = gelu_bf16(x)

        # Second conv (stride=2) + GELU
        x = self.conv2(x)
        x = gelu_bf16(x)

        # x shape: (batch, d_model, seq_len)
        # Transpose to (batch, seq_len, d_model) for transformer
        x = x.transpose(0, 2, 1)

        seq_len = x.shape[1]
        max_pos = self.positional_embedding.shape[0]

        # Truncate to max positional embedding size if needed
        if seq_len > max_pos:
            x = x[:, :max_pos, :]
            seq_len = max_pos

        # Add positional embedding
        x = add_bf16(x, self.positional_embedding[:seq_len])

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        x = self.ln_post(x)

        return x

    def load_weights(self, state_dict: dict):
        """
        Load weights from HuggingFace state dict.

        Args:
            state_dict: Dictionary of weight tensors
        """
        # Conv weights
        self.conv1.load_weights(
            state_dict["conv1.weight"],
            state_dict["conv1.bias"],
        )
        self.conv2.load_weights(
            state_dict["conv2.weight"],
            state_dict["conv2.bias"],
        )

        # Positional embedding
        self.positional_embedding = state_dict["embed_positions.weight"].astype(BF16_DTYPE)

        # Layer weights
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."

            # Layer norms
            layer.attn_ln.load_weights(
                state_dict[prefix + "self_attn_layer_norm.weight"],
                state_dict[prefix + "self_attn_layer_norm.bias"],
            )
            layer.mlp_ln.load_weights(
                state_dict[prefix + "final_layer_norm.weight"],
                state_dict[prefix + "final_layer_norm.bias"],
            )

            # Self-attention
            layer.self_attn.load_weights(
                q_proj_weight=state_dict[prefix + "self_attn.q_proj.weight"],
                k_proj_weight=state_dict[prefix + "self_attn.k_proj.weight"],
                v_proj_weight=state_dict[prefix + "self_attn.v_proj.weight"],
                out_proj_weight=state_dict[prefix + "self_attn.out_proj.weight"],
                out_proj_bias=state_dict[prefix + "self_attn.out_proj.bias"],
            )

            # MLP
            layer.mlp.load_weights(
                fc1_weight=state_dict[prefix + "fc1.weight"],
                fc1_bias=state_dict[prefix + "fc1.bias"],
                fc2_weight=state_dict[prefix + "fc2.weight"],
                fc2_bias=state_dict[prefix + "fc2.bias"],
            )

        # Final layer norm
        self.ln_post.load_weights(
            state_dict["layer_norm.weight"],
            state_dict["layer_norm.bias"],
        )
