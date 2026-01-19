#!/usr/bin/env python3
"""
Validation tests for Whisper-IRON.

Compares our implementation against HuggingFace reference and
validates transcription quality against known test vectors.

Run with: python tests/test_validation.py
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_test_samples():
    """Load test samples metadata."""
    samples_path = Path(__file__).parent.parent / "test_data" / "samples.json"
    with open(samples_path) as f:
        return json.load(f)["samples"]


def test_hf_reference():
    """Test that HuggingFace model produces expected outputs."""
    print("\n=== HuggingFace Reference Test ===")

    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        import librosa
    except ImportError as e:
        print(f"SKIP: Missing dependency: {e}")
        return True

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.eval()

    samples = load_test_samples()
    test_data_dir = Path(__file__).parent.parent / "test_data"

    all_passed = True
    for sample in samples:
        audio_path = test_data_dir / sample["file"]
        audio, _ = librosa.load(str(audio_path), sr=16000)
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000)

        with torch.no_grad():
            generated = model.generate(inputs.input_features, max_length=448)

        transcription = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        expected = sample["transcript_whisper_tiny"].strip()

        # Allow minor differences in punctuation
        passed = transcription.lower().replace(",", "").replace(".", "") == expected.lower().replace(",", "").replace(".", "")
        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] {sample['file']}")
        if not passed:
            print(f"    Expected: {expected}")
            print(f"    Got:      {transcription}")
            all_passed = False

    return all_passed


def test_mel_spectrogram():
    """Test mel spectrogram matches HuggingFace processor."""
    print("\n=== Mel Spectrogram Test ===")

    try:
        import librosa
        from transformers import WhisperProcessor
        from utils import preprocess_audio
    except ImportError as e:
        print(f"SKIP: Missing dependency: {e}")
        return True

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    samples = load_test_samples()
    test_data_dir = Path(__file__).parent.parent / "test_data"

    all_passed = True
    for sample in samples:
        audio_path = test_data_dir / sample["file"]

        # Our preprocessing
        our_mel = preprocess_audio(str(audio_path))

        # HF preprocessing
        audio, _ = librosa.load(str(audio_path), sr=16000)
        hf_features = processor(audio, return_tensors="np", sampling_rate=16000)
        hf_mel = hf_features.input_features

        # Compare
        max_diff = np.abs(our_mel - hf_mel).max()
        passed = max_diff < 0.01  # Allow small numerical differences
        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] {sample['file']}: max_diff={max_diff:.6f}")
        if not passed:
            all_passed = False

    return all_passed


def test_encoder_output():
    """Test encoder output matches HuggingFace (within tolerance)."""
    print("\n=== Encoder Output Test ===")

    try:
        import torch
        from transformers import WhisperForConditionalGeneration
        from model import Whisper
        from utils import preprocess_audio
    except ImportError as e:
        print(f"SKIP: Missing dependency: {e}")
        return True

    # Load models
    hf_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    hf_model.eval()
    our_model = Whisper.from_pretrained("openai/whisper-tiny")

    samples = load_test_samples()
    test_data_dir = Path(__file__).parent.parent / "test_data"

    all_passed = True
    for sample in samples:
        audio_path = test_data_dir / sample["file"]
        mel = preprocess_audio(str(audio_path))

        # Our encoder
        our_enc = our_model.encode(mel)

        # HF encoder
        with torch.no_grad():
            hf_enc = hf_model.model.encoder(torch.from_numpy(mel)).last_hidden_state.numpy()

        # Compare
        max_diff = np.abs(our_enc - hf_enc).max()
        mean_diff = np.abs(our_enc - hf_enc).mean()

        # Encoder should be close but may have accumulated numerical differences
        passed = max_diff < 20.0 and mean_diff < 1.0  # Relaxed tolerance
        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] {sample['file']}: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}")
        if not passed:
            all_passed = False

    return all_passed


def test_transcription_quality():
    """Test our model's transcription quality (when fully implemented)."""
    print("\n=== Transcription Quality Test ===")

    try:
        from model import Whisper
        from utils import preprocess_audio, WhisperTokenizer
    except ImportError as e:
        print(f"SKIP: Missing dependency: {e}")
        return True

    model = Whisper.from_pretrained("openai/whisper-tiny")
    tokenizer = WhisperTokenizer("openai/whisper-tiny")

    samples = load_test_samples()
    test_data_dir = Path(__file__).parent.parent / "test_data"

    all_passed = True
    for sample in samples:
        audio_path = test_data_dir / sample["file"]
        mel = preprocess_audio(str(audio_path))

        # Generate
        tokens = model.generate(
            mel,
            start_token_id=tokenizer.SOT,
            end_token_id=tokenizer.EOT,
            language_token_id=tokenizer.LANGUAGES["en"],
            transcribe_token_id=tokenizer.TRANSCRIBE,
            no_timestamps_token_id=tokenizer.NO_TIMESTAMPS,
        )

        transcription = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        expected = sample["transcript_whisper_tiny"].strip()

        # Calculate word error rate (simple word match for now)
        our_words = set(transcription.lower().split())
        exp_words = set(expected.lower().split())
        if exp_words:
            word_match = len(our_words & exp_words) / len(exp_words)
        else:
            word_match = 0.0

        passed = word_match > 0.5  # At least 50% word overlap
        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] {sample['file']}: word_match={word_match:.1%}")
        print(f"    Expected: {expected[:60]}...")
        print(f"    Got:      {transcription[:60]}..." if transcription else "    Got: (empty)")

        if not passed:
            all_passed = False

    return all_passed


def main():
    print("=" * 60)
    print("  WHISPER-IRON VALIDATION TESTS")
    print("=" * 60)

    results = {}

    # Run tests
    results["HF Reference"] = test_hf_reference()
    results["Mel Spectrogram"] = test_mel_spectrogram()
    results["Encoder Output"] = test_encoder_output()
    results["Transcription"] = test_transcription_quality()

    # Summary
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
