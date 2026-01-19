"""
Audio processing utilities for Whisper.

Converts audio files to mel spectrograms for model input.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, List

# Whisper audio parameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30  # seconds

# Supported audio formats (based on librosa/soundfile support)
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma', '.aac'}


class AudioLoadError(Exception):
    """Raised when audio file cannot be loaded."""
    pass


def _validate_audio_path(path: str) -> Path:
    """
    Validate audio file path exists and has supported format.

    Args:
        path: Path to audio file

    Returns:
        Validated Path object

    Raises:
        AudioLoadError: If file doesn't exist or format unsupported
    """
    audio_path = Path(path)

    if not audio_path.exists():
        raise AudioLoadError(f"Audio file not found: {path}")

    if not audio_path.is_file():
        raise AudioLoadError(f"Path is not a file: {path}")

    suffix = audio_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise AudioLoadError(
            f"Unsupported audio format '{suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    return audio_path


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file
        sr: Target sample rate (default: 16000 for Whisper)

    Returns:
        Audio waveform as 1D numpy array

    Raises:
        AudioLoadError: If file cannot be loaded
        ImportError: If librosa is not installed
    """
    # Validate path first
    audio_path = _validate_audio_path(path)

    try:
        import librosa
    except ImportError:
        raise ImportError(
            "librosa is required for audio loading. "
            "Install with: pip install librosa"
        )

    try:
        # Load and resample
        audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    except Exception as e:
        raise AudioLoadError(
            f"Failed to load audio file '{path}': {type(e).__name__}: {e}"
        ) from e

    if len(audio) == 0:
        raise AudioLoadError(f"Audio file is empty or contains no audio data: {path}")

    return audio.astype(np.float32)


def pad_or_trim(audio: np.ndarray, length: int = CHUNK_LENGTH * SAMPLE_RATE) -> np.ndarray:
    """
    Pad or trim audio to exact length.

    Args:
        audio: Audio waveform
        length: Target length in samples (default: 30 seconds)

    Returns:
        Audio of exact length
    """
    if len(audio) > length:
        audio = audio[:length]
    elif len(audio) < length:
        audio = np.pad(audio, (0, length - len(audio)))
    return audio


def compute_mel_spectrogram(
    audio: np.ndarray,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Compute log-mel spectrogram from audio waveform.

    This matches Whisper's preprocessing exactly.

    Args:
        audio: Audio waveform of shape (n_samples,)
        n_mels: Number of mel frequency bins
        n_fft: FFT window size
        hop_length: Hop length between frames
        sr: Sample rate

    Returns:
        Log-mel spectrogram of shape (n_mels, n_frames)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError(
            "librosa is required for mel spectrogram computation. "
            "Install with: pip install librosa"
        )

    # Compute STFT
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window='hann',
        center=True,
        pad_mode='reflect',
    )

    # Compute magnitude spectrogram
    magnitudes = np.abs(stft) ** 2

    # Create mel filterbank
    mel_filters = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0,
        fmax=sr / 2,
    )

    # Apply mel filterbank
    mel_spec = mel_filters @ magnitudes

    # Convert to log scale (with small epsilon for numerical stability)
    log_mel = np.log10(np.maximum(mel_spec, 1e-10))

    # Normalize (Whisper uses specific normalization)
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    # Whisper expects exactly 3000 frames for 30s audio
    # Trim to 3000 if we have slightly more due to STFT padding
    target_frames = 3000
    if log_mel.shape[1] > target_frames:
        log_mel = log_mel[:, :target_frames]

    return log_mel.astype(np.float32)


def preprocess_audio(
    audio_path: str,
    chunk_length: int = CHUNK_LENGTH,
) -> np.ndarray:
    """
    Full audio preprocessing pipeline.

    Args:
        audio_path: Path to audio file
        chunk_length: Audio chunk length in seconds

    Returns:
        Mel spectrogram ready for model input, shape (1, n_mels, n_frames)
    """
    # Load audio
    audio = load_audio(audio_path)

    # Pad or trim to chunk length
    target_length = chunk_length * SAMPLE_RATE
    audio = pad_or_trim(audio, target_length)

    # Compute mel spectrogram
    mel = compute_mel_spectrogram(audio)

    # Add batch dimension
    mel = mel[np.newaxis, :, :]

    return mel


def preprocess_audio_chunks(
    audio_path: str,
    chunk_length: int = CHUNK_LENGTH,
    overlap: float = 0.0,
) -> list:
    """
    Preprocess long audio into overlapping chunks.

    Args:
        audio_path: Path to audio file
        chunk_length: Chunk length in seconds
        overlap: Overlap between chunks (0.0 to 1.0)

    Returns:
        List of mel spectrograms, each of shape (1, n_mels, n_frames)
    """
    # Load full audio
    audio = load_audio(audio_path)

    # Calculate chunk parameters
    chunk_samples = chunk_length * SAMPLE_RATE
    hop_samples = int(chunk_samples * (1 - overlap))

    chunks = []
    start = 0

    while start < len(audio):
        end = start + chunk_samples
        chunk = audio[start:end]

        # Pad last chunk if needed
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        # Compute mel spectrogram
        mel = compute_mel_spectrogram(chunk)
        mel = mel[np.newaxis, :, :]
        chunks.append(mel)

        start += hop_samples

        # Stop if we've processed all audio
        if end >= len(audio):
            break

    return chunks
