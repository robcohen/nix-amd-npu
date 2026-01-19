#!/usr/bin/env python3
"""
Test IRON API functionality.

Tests that the IRON API is properly installed and can generate MLIR.
Does not require actual NPU hardware - just verifies the toolchain.

Usage:
    python kernels/npu/test_iron.py
"""

import sys


def test_iron_import():
    """Test that IRON modules can be imported."""
    print("Testing IRON imports...")

    try:
        from aie.iron import ObjectFifo, Worker, Runtime, Program
        print("  [OK] aie.iron core imports")
    except ImportError as e:
        print(f"  [FAIL] aie.iron: {e}")
        return False

    try:
        from aie.iron.device import NPU2
        print("  [OK] aie.iron.device")
    except ImportError as e:
        print(f"  [FAIL] aie.iron.device: {e}")
        return False

    try:
        from aie.iron.placers import SequentialPlacer
        print("  [OK] aie.iron.placers")
    except ImportError as e:
        print(f"  [FAIL] aie.iron.placers: {e}")
        return False

    try:
        from aie.iron.controlflow import range_
        print("  [OK] aie.iron.controlflow")
    except ImportError as e:
        print(f"  [FAIL] aie.iron.controlflow: {e}")
        return False

    try:
        import aie.iron as iron
        iron.set_current_device(NPU2())
        print("  [OK] aie.iron.set_current_device")
    except Exception as e:
        print(f"  [FAIL] set_current_device: {e}")
        return False

    return True


def test_simple_program():
    """Test generating a simple IRON program using new API."""
    print("\nTesting simple IRON program generation...")

    try:
        import numpy as np
        import aie.iron as iron
        from aie.iron import ObjectFifo, Program, Runtime, Worker
        from aie.iron.placers import SequentialPlacer
        from aie.iron.device import NPU2
        from aie.iron.controlflow import range_
    except ImportError as e:
        print(f"  [SKIP] Missing import: {e}")
        return True  # Not a failure, just skip

    try:
        # Set device
        iron.set_current_device(NPU2())

        # Simple passthrough program using new API
        @iron.jit(is_placed=False)
        def simple_passthrough(input_tensor, output_tensor):
            tile_size = 64
            num_elements = np.size(input_tensor)
            n_tiles = num_elements // tile_size
            dtype = input_tensor.dtype

            # Types
            tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
            tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]

            # Data flow
            of_in = ObjectFifo(tile_ty, name="in")
            of_out = ObjectFifo(tile_ty, name="out")

            # Worker: copy input to output
            def passthrough_fn(of_in, of_out):
                for _ in range_(n_tiles):
                    elem_in = of_in.acquire(1)
                    elem_out = of_out.acquire(1)
                    for i in range_(tile_size):
                        elem_out[i] = elem_in[i]
                    of_in.release(1)
                    of_out.release(1)

            worker = Worker(passthrough_fn, fn_args=[of_in.cons(), of_out.prod()])

            # Runtime sequence
            rt = Runtime()
            with rt.sequence(tensor_ty, tensor_ty) as (A, B):
                rt.start(worker)
                rt.fill(of_in.prod(), A)
                rt.drain(of_out.cons(), B, wait=True)

            # Return MLIR module
            return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())

        # Test with dummy tensors (JIT will generate MLIR without executing)
        input_t = iron.zeros((256,), dtype=np.int32)
        output_t = iron.zeros((256,), dtype=np.int32)

        # This should generate MLIR
        mlir_module = simple_passthrough(input_t, output_t)

        print(f"  [OK] Generated MLIR module")
        return True

    except ModuleNotFoundError as e:
        if "pyxrt" in str(e):
            print(f"  [SKIP] pyxrt not available (no NPU driver/hardware)")
            return True  # Not a failure, just no hardware
        print(f"  [FAIL] Program generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"  [FAIL] Program generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_objectfifo():
    """Test ObjectFifo creation."""
    print("\nTesting ObjectFifo creation...")

    try:
        import numpy as np
        from aie.iron import ObjectFifo

        tile_ty = np.ndarray[(1024,), np.dtype[np.float16]]
        of = ObjectFifo(tile_ty, name="test_fifo")

        print(f"  [OK] Created ObjectFifo: {of}")
        return True

    except Exception as e:
        print(f"  [FAIL] ObjectFifo: {e}")
        return False


def main():
    print("=" * 60)
    print("  IRON API Test Suite")
    print("=" * 60)

    results = []

    results.append(("IRON Imports", test_iron_import()))
    results.append(("ObjectFifo", test_objectfifo()))
    results.append(("Simple Program", test_simple_program()))

    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
