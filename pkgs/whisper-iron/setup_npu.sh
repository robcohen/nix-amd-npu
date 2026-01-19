#!/bin/bash
# Setup script for Whisper-IRON NPU development
#
# This script installs all required packages for NPU kernel development
# and testing. Run this INSIDE the iron-fhs environment.
#
# Usage:
#   nix develop .#iron-full
#   iron-fhs
#   cd pkgs/whisper-iron
#   ./setup_npu.sh

set -e

echo "========================================"
echo "  Whisper-IRON NPU Setup"
echo "========================================"

# Check if we're in a virtualenv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: No virtualenv detected. Creating one..."
    python -m venv .venv
    source .venv/bin/activate
fi

echo ""
echo "Step 1: Installing IRON packages..."
echo "----------------------------------------"

# Install mlir_aie and llvm-aie
pip install --upgrade pip
pip install mlir_aie llvm-aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels

# Install eudsl-python-extras with correct env var
EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX=aie pip install eudsl-python-extras==0.1.0.20251215.1800+3c7ac1b -f https://llvm.github.io/eudsl

echo ""
echo "Step 2: Installing Whisper dependencies..."
echo "----------------------------------------"
pip install transformers librosa numpy

echo ""
echo "Step 3: Verifying IRON installation..."
echo "----------------------------------------"
python -c "
from aie.iron import ObjectFifo, Worker, Runtime
print('  [OK] aie.iron imports successful')
" || echo "  [FAIL] IRON import failed"

echo ""
echo "Step 4: Checking kernel status..."
echo "----------------------------------------"
python -c "
from kernels.npu import get_kernel_status
import json
status = get_kernel_status()
for name, info in status.items():
    src = '✓' if info['source_exists'] else '✗'
    wrap = '✓' if info['wrapper_exists'] else '✗'
    comp = '✓' if info['compiled'] else '✗'
    print(f'  {name}: source={src} wrapper={wrap} compiled={comp}')
"

echo ""
echo "Step 5: Building C++ kernels..."
echo "----------------------------------------"
python kernels/npu/build.py || echo "  [WARN] Some kernels failed to compile"

echo ""
echo "Step 6: Running IRON API tests..."
echo "----------------------------------------"
python kernels/npu/test_iron.py

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run full NPU test suite:"
echo "     python test_npu.py --verbose"
echo ""
echo "  2. Run benchmarks:"
echo "     python benchmark.py --all"
echo ""
echo "  3. Test individual kernels:"
echo "     python kernels/npu/vector_add.py --verify"
echo "     python kernels/npu/matmul_kernel.py --verify --m 64 --k 64 --n 64"
echo ""
echo "  4. Transcribe audio:"
echo "     python transcribe.py test_data/1089-134686-0000.flac"
echo ""
