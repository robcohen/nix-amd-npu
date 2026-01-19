#!/usr/bin/env python3
"""Test VitisAI Execution Provider availability and functionality."""

import os
import sys

def check_onnxruntime():
    """Check basic ONNX Runtime installation."""
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
        return ort
    except ImportError as e:
        print(f"ONNX Runtime not installed: {e}")
        return None

def check_providers(ort):
    """Check available execution providers."""
    providers = ort.get_available_providers()
    print(f"\nAvailable providers: {providers}")

    has_vitisai = 'VitisAIExecutionProvider' in providers
    print(f"VitisAI EP available: {has_vitisai}")

    return has_vitisai

def check_environment():
    """Check environment variables."""
    print("\nEnvironment:")
    for var in ['XILINX_XRT', 'XLNX_VART_FIRMWARE', 'VAIP_CONFIG', 'LD_LIBRARY_PATH']:
        val = os.environ.get(var, '<not set>')
        # Truncate long values
        if len(val) > 80:
            val = val[:77] + "..."
        print(f"  {var}: {val}")

def check_xrt():
    """Check XRT status."""
    import subprocess
    try:
        result = subprocess.run(['xrt-smi', 'examine'], capture_output=True, text=True, timeout=10)
        print("\nXRT Status:")
        for line in result.stdout.split('\n')[:15]:  # First 15 lines
            print(f"  {line}")
        return result.returncode == 0
    except FileNotFoundError:
        print("\nxrt-smi not found in PATH")
        return False
    except Exception as e:
        print(f"\nXRT check failed: {e}")
        return False

def test_simple_inference(ort):
    """Test simple ONNX model with VitisAI EP."""
    import numpy as np

    # Create a minimal ONNX model for testing
    try:
        import onnx
        from onnx import helper, TensorProto

        # Simple identity model
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 10])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 10])

        identity_node = helper.make_node('Identity', ['X'], ['Y'])
        graph = helper.make_graph([identity_node], 'test', [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])

        model_path = '/tmp/test_identity.onnx'
        onnx.save(model, model_path)
        print(f"\nCreated test model: {model_path}")

    except Exception as e:
        print(f"\nFailed to create test model: {e}")
        return False

    # Try to load with VitisAI EP
    print("\nAttempting to load model with VitisAI EP...")
    try:
        sess_options = ort.SessionOptions()

        # Try with VitisAI EP
        session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['VitisAIExecutionProvider', 'CPUExecutionProvider']
        )

        # Check which provider is actually being used
        actual_providers = session.get_providers()
        print(f"Session providers: {actual_providers}")

        # Run inference
        input_data = np.random.randn(1, 10).astype(np.float32)
        outputs = session.run(None, {'X': input_data})

        # Verify output
        np.testing.assert_allclose(outputs[0], input_data, rtol=1e-5)
        print("Inference test: PASSED")

        return 'VitisAIExecutionProvider' in actual_providers

    except Exception as e:
        print(f"VitisAI EP load failed: {e}")
        print("\nTrying CPU-only fallback...")

        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_data = np.random.randn(1, 10).astype(np.float32)
            outputs = session.run(None, {'X': input_data})
            print("CPU fallback: PASSED")
        except Exception as e2:
            print(f"CPU fallback also failed: {e2}")

        return False

def main():
    print("=" * 60)
    print("VitisAI Execution Provider Test")
    print("=" * 60)

    check_environment()

    ort = check_onnxruntime()
    if not ort:
        return 1

    has_vitisai = check_providers(ort)
    check_xrt()

    if has_vitisai:
        using_npu = test_simple_inference(ort)
        if using_npu:
            print("\n" + "=" * 60)
            print("SUCCESS: VitisAI EP is working!")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("PARTIAL: VitisAI EP listed but not used for inference")
            print("This usually means the VAIP runtime library is missing")
            print("=" * 60)
            return 1
    else:
        print("\n" + "=" * 60)
        print("VitisAI EP not available in this ONNX Runtime build")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
