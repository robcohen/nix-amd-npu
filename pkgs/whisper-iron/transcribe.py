#!/usr/bin/env python3
"""
Whisper transcription CLI.

Usage:
    python transcribe.py audio.mp3
    python transcribe.py audio.wav --model openai/whisper-base
    python transcribe.py audio.flac --language en --task transcribe
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Whisper on AMD NPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s audio.mp3
    %(prog)s audio.wav --model openai/whisper-base
    %(prog)s podcast.mp3 --language en --output transcript.txt
        """,
    )

    parser.add_argument(
        "audio",
        type=str,
        help="Path to audio file (mp3, wav, flac, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-tiny",
        help="Model name (default: openai/whisper-tiny)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task: transcribe or translate to English (default: transcribe)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: print to stdout)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print timing and debug information",
    )

    args = parser.parse_args()

    # Check audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Import here to show errors after arg parsing
    from model import Whisper
    from utils import preprocess_audio, WhisperTokenizer

    # Load model
    if args.verbose:
        print(f"Loading model: {args.model}")
        start = time.time()

    model = Whisper.from_pretrained(args.model)

    if args.verbose:
        print(f"Model loaded in {time.time() - start:.2f}s")

    # Load tokenizer
    tokenizer = WhisperTokenizer(args.model)

    # Preprocess audio
    if args.verbose:
        print(f"Processing audio: {audio_path}")
        start = time.time()

    mel = preprocess_audio(str(audio_path))

    if args.verbose:
        print(f"Audio processed in {time.time() - start:.2f}s")
        print(f"Mel spectrogram shape: {mel.shape}")

    # Generate transcription
    if args.verbose:
        print("Generating transcription...")
        start = time.time()

    # Get prompt tokens based on settings
    prompt_tokens = tokenizer.get_prompt_tokens(
        language=args.language,
        task=args.task,
        timestamps=False,
    )

    # Run inference
    tokens = model.generate(
        mel,
        start_token_id=tokenizer.SOT,
        end_token_id=tokenizer.EOT,
        language_token_id=tokenizer.LANGUAGES.get(args.language, tokenizer.LANGUAGES["en"]),
        transcribe_token_id=tokenizer.TRANSCRIBE if args.task == "transcribe" else tokenizer.TRANSLATE,
        no_timestamps_token_id=tokenizer.NO_TIMESTAMPS,
    )

    if args.verbose:
        print(f"Generation completed in {time.time() - start:.2f}s")
        print(f"Generated {len(tokens)} tokens")

    # Decode tokens to text
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Output result
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(text)
        print(f"Transcription saved to: {output_path}")
    else:
        print("\n" + "=" * 50)
        print("TRANSCRIPTION:")
        print("=" * 50)
        print(text)
        print("=" * 50)


if __name__ == "__main__":
    main()
