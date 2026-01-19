"""Whisper model configuration for whisper-tiny."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WhisperConfig:
    """Configuration for Whisper model."""

    # Model architecture
    d_model: int = 384
    n_heads: int = 6
    head_dim: int = 64  # d_model // n_heads
    encoder_layers: int = 4
    decoder_layers: int = 4

    # Feed-forward network
    d_ff: int = 1536  # 4 * d_model

    # Vocabulary and sequence lengths
    vocab_size: int = 51865
    max_source_positions: int = 1500  # 30 seconds of audio
    max_target_positions: int = 448

    # Audio processing
    n_mels: int = 80
    n_audio_ctx: int = 1500

    # Conv layers in encoder
    encoder_conv_kernel_size: int = 3
    encoder_conv_channels: int = 384

    # Activation
    activation: str = "gelu"

    # Precision
    dtype: str = "bf16"


# Pre-defined configurations
WHISPER_TINY = WhisperConfig()

WHISPER_BASE = WhisperConfig(
    d_model=512,
    n_heads=8,
    head_dim=64,
    encoder_layers=6,
    decoder_layers=6,
    d_ff=2048,
    encoder_conv_channels=512,
)

WHISPER_SMALL = WhisperConfig(
    d_model=768,
    n_heads=12,
    head_dim=64,
    encoder_layers=12,
    decoder_layers=12,
    d_ff=3072,
    encoder_conv_channels=768,
)
