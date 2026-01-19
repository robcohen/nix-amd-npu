#!/usr/bin/env python3
"""
Export Whisper model to ONNX format for NPU inference.

This exports the encoder and decoder separately for optimization.
The encoder processes audio once, decoder runs autoregressively.
"""

import torch
import numpy as np
from pathlib import Path
import argparse

def export_whisper_encoder(model_name="openai/whisper-tiny", output_dir="onnx_models"):
    """Export Whisper encoder to ONNX."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print(f"Loading {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Export encoder
    print("Exporting encoder...")

    # Dummy input: mel spectrogram (batch=1, n_mels=80, frames=3000)
    dummy_mel = torch.randn(1, 80, 3000)

    # Get encoder
    encoder = model.get_encoder()

    # Trace and export
    torch.onnx.export(
        encoder,
        dummy_mel,
        output_dir / "whisper_encoder.onnx",
        input_names=["mel_spectrogram"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "mel_spectrogram": {2: "frames"},
            "encoder_hidden_states": {1: "sequence_length"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Saved: {output_dir}/whisper_encoder.onnx")

    # Check model size
    encoder_size = (output_dir / "whisper_encoder.onnx").stat().st_size / 1024 / 1024
    print(f"  Size: {encoder_size:.1f} MB")

    return output_dir / "whisper_encoder.onnx"


def export_whisper_decoder(model_name="openai/whisper-tiny", output_dir="onnx_models"):
    """Export Whisper decoder to ONNX (more complex due to KV cache)."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print(f"Loading {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Export decoder is more complex due to:
    # 1. Cross-attention to encoder states
    # 2. KV cache for efficient autoregressive generation
    # 3. Causal masking

    print("Exporting decoder (simplified, no KV cache)...")

    # For NPU, we might want to export without KV cache and handle it differently
    # This is a simplified export

    d_model = model.config.d_model  # 384 for tiny
    n_layers = model.config.decoder_layers  # 4 for tiny

    # Dummy inputs
    batch_size = 1
    encoder_seq_len = 1500
    decoder_seq_len = 1

    dummy_decoder_input_ids = torch.tensor([[50258]])  # Start token
    dummy_encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, d_model)

    # Get decoder
    decoder = model.get_decoder()

    try:
        torch.onnx.export(
            decoder,
            (dummy_decoder_input_ids, None, dummy_encoder_hidden_states),
            output_dir / "whisper_decoder.onnx",
            input_names=["input_ids", "attention_mask", "encoder_hidden_states"],
            output_names=["decoder_hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "encoder_hidden_states": {0: "batch", 1: "encoder_sequence"},
                "decoder_hidden_states": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  Saved: {output_dir}/whisper_decoder.onnx")

        decoder_size = (output_dir / "whisper_decoder.onnx").stat().st_size / 1024 / 1024
        print(f"  Size: {decoder_size:.1f} MB")
    except Exception as e:
        print(f"  Decoder export failed: {e}")
        print("  (Decoder export is complex due to cross-attention)")

    return output_dir


def check_onnx_model(onnx_path):
    """Validate ONNX model."""
    import onnx

    print(f"\nValidating {onnx_path}...")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("  ONNX model is valid!")

    # Print model info
    print(f"  Inputs: {[i.name for i in model.graph.input]}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")
    print(f"  Nodes: {len(model.graph.node)}")

    # Count ops
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"  Top ops: {sorted(ops.items(), key=lambda x: -x[1])[:10]}")


def main():
    parser = argparse.ArgumentParser(description="Export Whisper to ONNX")
    parser.add_argument("--model", default="openai/whisper-tiny")
    parser.add_argument("--output", default="onnx_models")
    parser.add_argument("--encoder-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Whisper ONNX Export")
    print("=" * 60)

    # Export encoder
    encoder_path = export_whisper_encoder(args.model, args.output)
    check_onnx_model(encoder_path)

    if not args.encoder_only:
        # Export decoder (may fail due to complexity)
        export_whisper_decoder(args.model, args.output)

    print("\n" + "=" * 60)
    print("ONNX Export Complete")
    print("=" * 60)
    print(f"""
Next steps for NPU inference:

1. QUANTIZE the model for INT8/BF16:
   - Use AMD Quark or Vitis AI Quantizer
   - Or use ONNX Runtime's quantization tools

2. COMPILE for NPU:
   - Use VitisAI EP which compiles on first run
   - Compiled model is cached for subsequent runs

3. RUN INFERENCE:
   ```python
   import onnxruntime as ort
   session = ort.InferenceSession(
       "whisper_encoder.onnx",
       providers=['VitisAIExecutionProvider']
   )
   ```

Note: VitisAI EP requires AMD Ryzen AI Software.
      Currently Early Access for Linux.
""")


if __name__ == "__main__":
    main()
