#!/usr/bin/env python3
"""
Fix for JIT overhead: Increase memory cache size.

The default IRON JIT only caches 1 kernel in memory. This means every
kernel switch requires reloading from disk (~50-100ms).

Usage:
    # At the start of your program:
    from kernels.npu.fix_cache import increase_cache_size
    increase_cache_size(100)  # Cache up to 100 kernels

    # Then use IRON normally
    from kernels.npu.gelu_kernel import gelu_iron
    ...
"""

def increase_cache_size(max_size: int = 100):
    """Increase the JIT memory cache size."""
    try:
        from aie.utils.jit import _compiled_kernels
        from aie.utils.compile.cache.circular_cache import CircularCache

        # Replace the cache with a larger one
        import aie.utils.jit as jit_module
        old_size = jit_module._compiled_kernels.max_size
        jit_module._compiled_kernels = CircularCache(max_size=max_size)
        print(f"JIT cache size increased: {old_size} -> {max_size}")
        return True
    except Exception as e:
        print(f"Failed to increase cache size: {e}")
        return False


def get_cache_stats():
    """Get current cache statistics."""
    try:
        from aie.utils.jit import _compiled_kernels
        from aie.utils.compile import NPU_CACHE_HOME
        import os

        disk_count = 0
        if os.path.exists(NPU_CACHE_HOME):
            disk_count = len([d for d in os.listdir(NPU_CACHE_HOME)
                             if os.path.isdir(NPU_CACHE_HOME / d)])

        return {
            "memory_cache_size": len(_compiled_kernels),
            "memory_cache_max": _compiled_kernels.max_size,
            "disk_cache_count": disk_count,
            "disk_cache_path": str(NPU_CACHE_HOME),
        }
    except Exception as e:
        return {"error": str(e)}


def prewarm_cache():
    """Pre-warm the cache by loading all kernels once."""
    import numpy as np

    print("Pre-warming NPU kernel cache...")

    # Import all kernels to trigger JIT compilation
    try:
        from kernels.npu.gelu_kernel import gelu_iron
        x = np.random.randn(1024).astype(np.float32)
        gelu_iron(x)
        print("  GELU: cached")
    except Exception as e:
        print(f"  GELU: failed ({e})")

    try:
        from kernels.npu.softmax_kernel import softmax_iron
        x = np.random.randn(16, 64).astype(np.float32)
        softmax_iron(x)
        print("  Softmax: cached")
    except Exception as e:
        print(f"  Softmax: failed ({e})")

    try:
        from kernels.npu.vector_add import vector_add_iron
        a = np.random.randn(1024).astype(np.float32)
        b = np.random.randn(1024).astype(np.float32)
        vector_add_iron(a, b)
        print("  VectorAdd: cached")
    except Exception as e:
        print(f"  VectorAdd: failed ({e})")

    print("Cache pre-warming complete")


if __name__ == "__main__":
    print("=== JIT Cache Fix ===\n")

    print("Before fix:")
    print(f"  {get_cache_stats()}")

    increase_cache_size(100)

    print("\nAfter fix:")
    print(f"  {get_cache_stats()}")
