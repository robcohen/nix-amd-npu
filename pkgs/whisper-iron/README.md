# Whisper-IRON: Speech Recognition on AMD Ryzen AI NPU

OpenAI Whisper implementation using MLIR-AIE/IRON for AMD Ryzen AI NPU acceleration.

## Quick Start

```bash
# Enter development shell
nix develop .#whisper

# Or use iron shell
nix develop .#iron
cd pkgs/whisper-iron

# Run NPU tests
python tests/test_npu.py

# Transcribe audio
python transcribe.py audio.wav --verbose
```

## Architecture

```
Audio File (.wav, .mp3, etc.)
         │
         ▼
┌──────────────────────────┐
│    Mel Spectrogram       │  CPU (librosa)
│    80 bins × 3000 frames │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│       ENCODER            │  NPU (when available)
│  ┌────────────────────┐  │
│  │ Conv1D (80→384)    │  │
│  │ Conv1D (384→384)   │  │
│  │ + Positional Embed │  │
│  └────────────────────┘  │
│  ┌────────────────────┐  │
│  │ Transformer × 4    │  │
│  │  • Self-Attention  │  │  ← matmul_bf16, softmax_bf16
│  │  • LayerNorm       │  │  ← layernorm_bf16
│  │  • MLP (GELU)      │  │  ← gelu_bf16
│  └────────────────────┘  │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│       DECODER            │  NPU (when available)
│  ┌────────────────────┐  │
│  │ Token Embedding    │  │
│  │ + Positional Embed │  │
│  └────────────────────┘  │
│  ┌────────────────────┐  │
│  │ Transformer × 4    │  │
│  │  • Causal Self-Attn│  │
│  │  • Cross-Attention │  │  ← attends to encoder
│  │  • LayerNorm       │  │
│  │  • MLP (GELU)      │  │
│  └────────────────────┘  │
│  ┌────────────────────┐  │
│  │ Output Projection  │  │  → vocab logits
│  └────────────────────┘  │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│    Greedy Decoding       │  CPU
│    + Tokenizer           │
└───────────┬──────────────┘
            │
            ▼
        Transcript
```

## NPU Integration

The kernels use MLIR-AIE to generate NPU-accelerated code:

### How It Works

1. **Kernel Definition**: Each operation (matmul, softmax, etc.) is defined using MLIR-AIE Python bindings
2. **Compilation**: `aiecc.py` compiles the MLIR to xclbin (NPU bitstream)
3. **Caching**: Compiled xclbin files are cached in `~/.cache/whisper-iron/`
4. **Execution**: At runtime, XRT loads the xclbin and handles DMA transfers

### Kernel Status

| Kernel | NPU Support | Notes |
|--------|-------------|-------|
| `matmul_bf16` | Yes | Tiled 64×64, auto-padding |
| `gelu_bf16` | Yes | Chunked processing |
| `softmax_bf16` | CPU | NPU version planned |
| `layernorm_bf16` | CPU | NPU version planned |
| `conv1d_bf16` | CPU | Uses im2col + matmul |

### Memory Constraints

- AIE tile memory: ~16KB per tile
- BF16 element: 2 bytes
- Max elements per tile: ~8192
- Tile size used: 64×64 = 4096 elements

Large matrices are automatically tiled and streamed through the NPU.

## Project Structure

```
whisper-iron/
├── kernels/               # NPU kernels
│   ├── npu_runtime.py     # MLIR-AIE compilation & XRT execution
│   ├── matmul.py          # Matrix multiplication (NPU-accelerated)
│   ├── elementwise.py     # GELU, ReLU, etc. (NPU-accelerated)
│   ├── softmax.py         # Softmax
│   ├── layernorm.py       # Layer normalization
│   └── conv1d.py          # 1D convolution
├── model/                 # Whisper architecture
│   ├── config.py          # Model configs (tiny/base/small)
│   ├── attention.py       # Multi-head & cross attention
│   ├── mlp.py             # Feed-forward block
│   ├── encoder.py         # Audio encoder
│   ├── decoder.py         # Text decoder
│   └── whisper.py         # Full model + generation
├── utils/
│   ├── audio.py           # Mel spectrogram (librosa)
│   └── tokenizer.py       # HuggingFace tokenizer
├── tests/
│   ├── test_kernels.py    # Kernel unit tests
│   └── test_npu.py        # NPU integration tests
├── transcribe.py          # CLI tool
└── requirements.txt
```

## Usage

### CLI

```bash
# Basic transcription
python transcribe.py audio.wav

# With options
python transcribe.py audio.mp3 \
    --model openai/whisper-tiny \
    --language en \
    --output transcript.txt \
    --verbose
```

### Python API

```python
from model import Whisper
from utils import preprocess_audio, WhisperTokenizer

# Load model (downloads from HuggingFace)
model = Whisper.from_pretrained("openai/whisper-tiny")
tokenizer = WhisperTokenizer("openai/whisper-tiny")

# Preprocess audio
mel = preprocess_audio("audio.wav")

# Generate transcription
tokens = model.generate(mel)
text = tokenizer.decode(tokens, skip_special_tokens=True)
print(text)
```

### Check NPU Status

```python
from kernels import npu_available

if npu_available():
    print("NPU acceleration enabled!")
else:
    print("Running on CPU (numpy fallback)")
```

## Model Configurations

| Model | Params | d_model | Layers | Heads |
|-------|--------|---------|--------|-------|
| tiny  | 39M    | 384     | 4+4    | 6     |
| base  | 74M    | 512     | 6+6    | 8     |
| small | 244M   | 768     | 12+12  | 12    |

## Performance

Benchmarks on AMD Ryzen AI 7 PRO 350 (Strix Point):

| Operation | Size | CPU (ms) | GFLOPS | Notes |
|-----------|------|----------|--------|-------|
| MatMul    | 384×384 | 0.54 | 210.66 | Linear projection |
| MatMul    | 1500×384 | 1.04 | 426.38 | Encoder attention |
| MatMul    | 448×384 | 0.84 | 157.15 | Decoder attention |
| Attention (full) | 6×100×64 | 5.48 | - | Q@K^T + softmax + @V |

*CPU baseline using numpy; NPU acceleration in development*

### Current Status

- **Kernel Tests**: All pass (CPU fallback mode)
- **NPU Detection**: XRT driver works, NPU hardware detected
- **MLIR-AIE**: MLIR bindings available, aiecc.py compiler accessible
- **IRON API**: Available via `nix develop .#iron-full` devshell

### NPU Kernel Development

Two options for developing NPU kernels:

**Option 1: iron-full devshell (Recommended)**
```bash
# Enter devshell
nix develop .#iron-full
iron-fhs

# First-time setup (inside FHS environment)
pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-2
pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX=aie pip install eudsl-python-extras==0.1.0.20251215.1800+3c7ac1b -f https://llvm.github.io/eudsl

# Test IRON
python -c 'from aie.iron import ObjectFifo; print("IRON works!")'
```

**Option 2: Basic iron shell**
```bash
nix develop .#iron  # Uses pre-built mlir-aie wheel (IRON API limited)
```

IRON kernels require C++ source files compiled with aiecc.py. See the
[mlir-aie programming examples](https://github.com/Xilinx/mlir-aie/tree/main/programming_examples)
for complete kernel implementations.

## Requirements

- AMD Ryzen AI processor (Phoenix/Hawk Point/Strix)
- Linux kernel 6.10+ with XDNA driver
- XRT (Xilinx Runtime)
- Python 3.10+

### Python Dependencies

```
numpy>=1.20.0
librosa>=0.9.0
transformers>=4.30.0
soundfile
pytest
mlir_aie  # From Xilinx releases
```

## Troubleshooting

### NPU Not Detected

```bash
# Check driver
xrt-smi examine

# Check XRT setup
echo $XILINX_XRT
ls $XILINX_XRT/lib/

# Check XDNA driver
lsmod | grep xdna
```

### Compilation Errors

```bash
# Check aiecc.py is available
which aiecc.py

# Clear kernel cache
rm -rf ~/.cache/whisper-iron/
```

### Model Download Issues

```bash
# Set HuggingFace cache
export HF_HOME=~/.cache/huggingface

# Or download manually
python -c "from transformers import WhisperForConditionalGeneration; WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny')"
```

## License

Apache 2.0

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [MLIR-AIE](https://github.com/Xilinx/mlir-aie)
- [AMD IRON](https://github.com/amd/IRON)
