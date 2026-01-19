#!/usr/bin/env python3
"""Check JIT caching and AOT options."""
import aie.iron as iron
import inspect

print("=== IRON JIT Options ===\n")

# Check jit decorator
print("iron.jit signature:")
print(f"  {inspect.signature(iron.jit)}")

# Check if there's a cache
print("\niron module attributes:")
for attr in dir(iron):
    if 'cache' in attr.lower() or 'compile' in attr.lower() or 'aot' in attr.lower():
        print(f"  {attr}")

# Check the jit decorator options
print("\njit decorator parameters:")
jit_sig = inspect.signature(iron.jit)
for param_name, param in jit_sig.parameters.items():
    print(f"  {param_name}: {param.default}")

# Check if there's mlir export
print("\nLooking for MLIR export...")
try:
    from aie.compiler import aiecc
    print("  aie.compiler.aiecc: FOUND")
except ImportError as e:
    print(f"  aie.compiler.aiecc: NOT FOUND ({e})")

try:
    from aie.dialects import aie as aie_dialect
    print("  aie.dialects.aie: FOUND")
except ImportError as e:
    print(f"  aie.dialects.aie: NOT FOUND ({e})")

# Check XRT runtime for loading xclbin
print("\nXRT runtime options:")
try:
    import pyxrt
    print("  pyxrt: FOUND")
    print(f"  pyxrt attributes: {[a for a in dir(pyxrt) if not a.startswith('_')][:10]}...")
except ImportError as e:
    print(f"  pyxrt: NOT FOUND ({e})")
