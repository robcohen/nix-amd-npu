#!/usr/bin/env python3
"""Analyze Whisper operations to find optimization opportunities."""
import numpy as np

print("=== Whisper-tiny Operation Analysis ===\n")

# Whisper-tiny dimensions
d_model = 384
n_heads = 6
head_dim = 64
n_encoder_layers = 4
n_decoder_layers = 4
max_audio_frames = 1500  # 30s audio
vocab_size = 51865

print("Model dimensions:")
print(f"  d_model: {d_model}")
print(f"  n_heads: {n_heads}")
print(f"  head_dim: {head_dim}")
print(f"  encoder_layers: {n_encoder_layers}")
print(f"  decoder_layers: {n_decoder_layers}")

# Count operations per forward pass
print("\n=== Operations per Forward Pass ===\n")

# Encoder (runs once per audio)
print("ENCODER (1x per audio):")
encoder_ops = []

# Conv1D layers (2)
encoder_ops.append(("Conv1D", 2, "80x3000 -> 384x1500"))

# Per encoder layer
for layer in range(n_encoder_layers):
    # Self-attention: Q, K, V projections
    encoder_ops.append((f"L{layer} Q/K/V proj", 3, f"matmul {d_model}x{d_model}"))
    encoder_ops.append((f"L{layer} Attention", 1, f"Q@K^T softmax @V"))
    encoder_ops.append((f"L{layer} Out proj", 1, f"matmul {d_model}x{d_model}"))
    encoder_ops.append((f"L{layer} LayerNorm", 2, f"normalize {d_model}"))
    # MLP
    encoder_ops.append((f"L{layer} MLP fc1", 1, f"matmul {d_model}x{d_model*4}"))
    encoder_ops.append((f"L{layer} GELU", 1, f"activation {d_model*4}"))
    encoder_ops.append((f"L{layer} MLP fc2", 1, f"matmul {d_model*4}x{d_model}"))

print(f"  Total encoder ops: {len(encoder_ops)}")

# Decoder (runs once per output token, ~50-100 tokens typical)
print("\nDECODER (Nx per output token):")
decoder_ops_per_token = []

for layer in range(n_decoder_layers):
    # Self-attention
    decoder_ops_per_token.append((f"L{layer} Self-attn Q/K/V", 3, f"matmul"))
    decoder_ops_per_token.append((f"L{layer} Self-attn", 1, f"attention"))
    decoder_ops_per_token.append((f"L{layer} Self-attn out", 1, f"matmul"))
    decoder_ops_per_token.append((f"L{layer} LayerNorm", 1, f"normalize"))
    # Cross-attention
    decoder_ops_per_token.append((f"L{layer} Cross-attn Q", 1, f"matmul"))
    decoder_ops_per_token.append((f"L{layer} Cross-attn K/V", 2, f"matmul (cached after first)"))
    decoder_ops_per_token.append((f"L{layer} Cross-attn", 1, f"attention"))
    decoder_ops_per_token.append((f"L{layer} Cross-attn out", 1, f"matmul"))
    decoder_ops_per_token.append((f"L{layer} LayerNorm", 1, f"normalize"))
    # MLP
    decoder_ops_per_token.append((f"L{layer} MLP fc1", 1, f"matmul"))
    decoder_ops_per_token.append((f"L{layer} GELU", 1, f"activation"))
    decoder_ops_per_token.append((f"L{layer} MLP fc2", 1, f"matmul"))
    decoder_ops_per_token.append((f"L{layer} LayerNorm", 1, f"normalize"))

# Output projection
decoder_ops_per_token.append(("LM head", 1, f"matmul {d_model}x{vocab_size}"))

print(f"  Ops per token: {len(decoder_ops_per_token)}")

# Estimate total kernel calls
n_output_tokens = 50  # typical transcription
total_kernel_calls = len(encoder_ops) + len(decoder_ops_per_token) * n_output_tokens
print(f"\n=== Total Kernel Calls ===")
print(f"  Encoder: {len(encoder_ops)}")
print(f"  Decoder: {len(decoder_ops_per_token)} x {n_output_tokens} tokens = {len(decoder_ops_per_token) * n_output_tokens}")
print(f"  TOTAL: {total_kernel_calls} kernel calls")

# Estimate overhead
kernel_overhead_ms = 2  # ms per kernel call with cache
total_overhead_ms = total_kernel_calls * kernel_overhead_ms
print(f"\n=== Overhead Estimate ===")
print(f"  Per-call overhead: {kernel_overhead_ms}ms")
print(f"  Total overhead: {total_overhead_ms/1000:.1f}s")
print(f"  CPU baseline: 17.3s")

print("\n=== Optimization Opportunities ===")
print("""
1. FUSED ATTENTION KERNEL
   - Combine Q@K^T, softmax, @V into single kernel
   - Saves: 3 kernel calls -> 1 per attention layer
   - Total savings: (4 encoder + 4*2 decoder) * 50 tokens = 600 calls

2. FUSED MLP KERNEL
   - Combine fc1, GELU, fc2 into single kernel
   - Saves: 3 kernel calls -> 1 per MLP
   - Total savings: (4 encoder + 4 decoder) * 50 tokens = 400 calls

3. KEEP DATA ON NPU
   - Avoid host-device sync between layers
   - Currently: sync after every kernel call
   - With fusion: sync only at layer boundaries

4. KV CACHE ON NPU
   - Keep decoder K/V cache on NPU memory
   - Avoid transferring 1500x384 tensors per token

ESTIMATED SPEEDUP WITH OPTIMIZATIONS:
  Current overhead: ~3000 kernel calls * 2ms = 6s
  Optimized: ~300 fused calls * 2ms = 0.6s
  Potential: 10x reduction in overhead
""")
