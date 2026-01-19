#!/usr/bin/env python3
"""Check NPU cache location and contents."""
import os

print("=== NPU Cache Location ===\n")

from aie.utils.compile import NPU_CACHE_HOME
print(f"NPU_CACHE_HOME: {NPU_CACHE_HOME}")
print(f"Exists: {os.path.exists(NPU_CACHE_HOME)}")

if os.path.exists(NPU_CACHE_HOME):
    print(f"\nCached kernels:")
    for item in os.listdir(NPU_CACHE_HOME):
        item_path = NPU_CACHE_HOME / item
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            xclbin_exists = "final.xclbin" in files
            size = 0
            if xclbin_exists:
                size = os.path.getsize(item_path / "final.xclbin") / 1024
            print(f"  {item[:16]}... - xclbin: {'YES' if xclbin_exists else 'NO'} ({size:.1f}KB)")

# Check memory cache size
print("\n=== Memory Cache ===")
from aie.utils.jit import _compiled_kernels
print(f"Cache type: {type(_compiled_kernels)}")
print(f"Max size: {_compiled_kernels.max_size}")
print(f"Current size: {len(_compiled_kernels)}")

# Suggest fix
print("\n=== Suggested Fix ===")
print("""
To reduce JIT overhead:

1. INCREASE MEMORY CACHE SIZE:
   Edit: .venv-iron/lib/python3.12/site-packages/mlir_aie/python/aie/utils/jit.py
   Change: _compiled_kernels = CircularCache(max_size=1)
   To:     _compiled_kernels = CircularCache(max_size=100)

2. PRE-WARM THE CACHE:
   Run each kernel once at startup with representative sizes.
   Subsequent calls will load from disk cache (~50ms vs ~500ms).

3. FIX KERNEL SIZES:
   Use fixed tile sizes (1024) to maximize cache hits.
   Avoid dynamic shapes that generate new kernels.
""")
