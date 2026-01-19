#!/usr/bin/env python3
"""Check aiecc compiler for AOT compilation."""
import inspect

print("=== AIECC Compiler Options ===\n")

try:
    from aie.compiler import aiecc
    print("aiecc module attributes:")
    for attr in dir(aiecc):
        if not attr.startswith('_'):
            obj = getattr(aiecc, attr)
            if callable(obj):
                try:
                    sig = inspect.signature(obj)
                    print(f"  {attr}{sig}")
                except:
                    print(f"  {attr}(...)")
            else:
                print(f"  {attr} = {type(obj).__name__}")
except Exception as e:
    print(f"Error: {e}")

# Check for MLIR module generation
print("\n=== MLIR Generation ===")
try:
    from aie.extras.context import mlir_mod_ctx
    print("mlir_mod_ctx: FOUND")
except ImportError as e:
    print(f"mlir_mod_ctx: NOT FOUND ({e})")

# Check iron internals for compilation
print("\n=== IRON Internals ===")
try:
    from aie.iron.program import program
    print("program module attrs:")
    for attr in dir(program):
        if not attr.startswith('_') and 'compile' in attr.lower():
            print(f"  {attr}")
except ImportError as e:
    print(f"program module: NOT FOUND ({e})")

# Check if there's a way to get xclbin from jit
print("\n=== JIT Internals ===")
try:
    from aie.iron import jit as jit_module
    print(f"jit module type: {type(jit_module)}")
    if hasattr(jit_module, '__wrapped__'):
        print(f"  wrapped: {jit_module.__wrapped__}")
except Exception as e:
    print(f"Error: {e}")

# Try to understand the compilation pipeline
print("\n=== Compilation Pipeline ===")
try:
    # Create a simple kernel and see what it produces
    import numpy as np
    import aie.iron as iron
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.placers import SequentialPlacer
    from aie.iron.controlflow import range_

    device = iron.get_current_device()

    @iron.jit(is_placed=False)
    def test_kernel(a, b, c):
        tensor_ty = np.ndarray[(16,), np.dtype[np.float32]]
        of_a = ObjectFifo(tensor_ty, name="A")
        of_c = ObjectFifo(tensor_ty, name="C")

        def core(of_a, of_c):
            ea = of_a.acquire(1)
            ec = of_c.acquire(1)
            for i in range_(16):
                ec[i] = ea[i] * 2.0
            of_a.release(1)
            of_c.release(1)

        worker = Worker(core, fn_args=[of_a.cons(), of_c.prod()])
        rt = Runtime()
        with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a_h, b_h, c_h):
            rt.start(worker)
            rt.fill(of_a.prod(), a_h)
            rt.drain(of_c.cons(), c_h, wait=True)

        return Program(device, rt).resolve_program(SequentialPlacer())

    # Get the compiled result
    a = iron.zeros((16,), dtype=np.float32)
    b = iron.zeros((16,), dtype=np.float32)
    c = iron.zeros((16,), dtype=np.float32)

    # This triggers compilation
    result = test_kernel(a, b, c)
    print(f"Kernel result type: {type(result)}")
    print(f"Kernel result: {result}")

    # Check if we can access the compiled artifact
    if hasattr(result, 'xclbin'):
        print("  Has xclbin attribute!")
    if hasattr(result, 'module'):
        print("  Has module attribute!")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
