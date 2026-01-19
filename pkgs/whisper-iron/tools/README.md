# Whisper-IRON Development Tools

This directory contains development, debugging, and benchmarking scripts for Whisper-IRON.
These are not part of the main package and are intended for developers working on NPU acceleration.

## Scripts

### Benchmarking

| Script | Description |
|--------|-------------|
| `benchmark.py` | General performance benchmarking for Whisper inference |
| `benchmark_npu.py` | NPU-specific benchmarking with detailed timing breakdown |

### Analysis

| Script | Description |
|--------|-------------|
| `analyze_whisper_ops.py` | Analyze Whisper model operations for NPU acceleration potential |
| `export_onnx.py` | Export Whisper model to ONNX format |

### Debugging

| Script | Description |
|--------|-------------|
| `check_aiecc.py` | Verify MLIR-AIE compiler (aiecc) installation |
| `check_aot.py` | Check ahead-of-time compilation setup |
| `check_cache_location.py` | Debug kernel cache directory issues |
| `check_jit_cache.py` | Inspect JIT compilation cache |
| `check_jit_internals.py` | Debug JIT compilation internals |
| `test_cache_fix.py` | Test fixes for cache-related issues |
| `test_vitisai_ep.py` | Test VitisAI execution provider |

## Usage

Run from the whisper-iron directory with the appropriate dev shell:

```bash
# Enter iron development shell
nix develop .#iron

# Run a tool
cd pkgs/whisper-iron
python tools/benchmark.py
```
