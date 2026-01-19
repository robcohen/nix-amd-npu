#!/usr/bin/env python3
"""Test if increasing cache size reduces overhead."""
import numpy as np
import time

print("=== Testing JIT Cache Fix ===\n")

# First, test WITHOUT the fix
print("1. Testing WITHOUT cache fix (max_size=1):")
from kernels.npu.gelu_kernel import gelu_iron
from kernels.npu.softmax_kernel import softmax_iron

x = np.random.randn(1024).astype(np.float32)

# Warmup
gelu_iron(x)
softmax_iron(x.reshape(1, -1))

# Measure alternating kernel calls
times = []
for i in range(5):
    start = time.perf_counter()
    gelu_iron(x)
    softmax_iron(x.reshape(1, -1))
    times.append(time.perf_counter() - start)

print(f"   Average time per pair: {np.mean(times)*1000:.1f}ms")
print(f"   Times: {[f'{t*1000:.1f}ms' for t in times]}")

# Now apply the fix
print("\n2. Applying cache fix (max_size=100):")
from kernels.npu.fix_cache import increase_cache_size, get_cache_stats
increase_cache_size(100)
print(f"   Stats: {get_cache_stats()}")

# Measure again
times_fixed = []
for i in range(5):
    start = time.perf_counter()
    gelu_iron(x)
    softmax_iron(x.reshape(1, -1))
    times_fixed.append(time.perf_counter() - start)

print(f"\n   Average time per pair: {np.mean(times_fixed)*1000:.1f}ms")
print(f"   Times: {[f'{t*1000:.1f}ms' for t in times_fixed]}")

# Improvement
improvement = (np.mean(times) - np.mean(times_fixed)) / np.mean(times) * 100
print(f"\n   Improvement: {improvement:.1f}%")

# Test many consecutive calls to same kernel
print("\n3. Testing consecutive calls to SAME kernel:")
times_single = []
for i in range(10):
    start = time.perf_counter()
    gelu_iron(x)
    times_single.append(time.perf_counter() - start)

print(f"   Average: {np.mean(times_single)*1000:.2f}ms")
print(f"   Min: {np.min(times_single)*1000:.2f}ms")
print(f"   Max: {np.max(times_single)*1000:.2f}ms")
