#!/usr/bin/env python3
"""Check JIT internals for AOT compilation."""
import os
import tempfile

print("=== JIT Internals ===\n")

# Check environment variables that might control compilation
print("Relevant environment variables:")
for key in sorted(os.environ.keys()):
    if any(x in key.lower() for x in ['aie', 'xrt', 'mlir', 'cache', 'compile']):
        print(f"  {key}={os.environ[key][:50]}...")

# Check the JIT source code location
print("\n=== JIT Source ===")
try:
    import aie.iron
    jit_file = aie.iron.jit.__code__.co_filename
    print(f"JIT defined in: {jit_file}")

    # Read the JIT source to understand it
    with open(jit_file) as f:
        lines = f.readlines()
        print(f"File has {len(lines)} lines")

        # Look for cache/xclbin related code
        for i, line in enumerate(lines):
            if any(x in line.lower() for x in ['xclbin', 'cache', 'save', 'load', 'compile']):
                print(f"  L{i+1}: {line.rstrip()[:80]}")

except Exception as e:
    print(f"Error: {e}")

# Check for XRT compilation artifacts
print("\n=== XRT Artifacts ===")
xrt_dirs = [
    "/tmp",
    os.path.expanduser("~/.cache"),
    os.path.expanduser("~/.xrt"),
]
for d in xrt_dirs:
    if os.path.exists(d):
        for f in os.listdir(d)[:20]:
            if 'xclbin' in f.lower() or 'aie' in f.lower():
                print(f"  {d}/{f}")

# Check IRON's hostruntime for xclbin handling
print("\n=== XRT Runtime ===")
try:
    from aie.utils.hostruntime import xrtruntime
    print(f"xrtruntime module: {xrtruntime}")
    for attr in dir(xrtruntime):
        if not attr.startswith('_'):
            print(f"  {attr}")
except Exception as e:
    print(f"Error: {e}")
