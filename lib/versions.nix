# Centralized version management for nix-amd-npu packages
#
# This file contains all version information and source hashes.
# Update versions here when upgrading packages.
#
# ============================================================================
# VERSION COMPATIBILITY MATRIX
# ============================================================================
#
# CRITICAL: XRT and XDNA driver versions MUST match exactly!
#   - XRT version format: YYYYMM.MAJOR.MINOR.PATCH (e.g., 202610.2.21.21)
#   - XDNA driver uses same version scheme
#   - Mismatched versions will cause runtime failures
#
# Component Dependencies:
#   ┌─────────────────┐
#   │   Applications  │  whisper-iron, user apps
#   ├─────────────────┤
#   │   MLIR-AIE      │  v1.1.4 (requires Python 3.12, cp312 wheel)
#   ├─────────────────┤
#   │  ONNX Runtime   │  VitisAI EP requires dynamic-dispatch
#   ├─────────────────┤
#   │    Vitis AI     │  unilog → xir → target-factory → vart
#   │                 │  graph-engine, dynamic-dispatch, xaiengine
#   ├─────────────────┤
#   │  XRT + XDNA     │  Must be same version (202610.2.21.21)
#   ├─────────────────┤
#   │  Linux Kernel   │  6.10+ required, 6.14+ has mainline amdxdna
#   └─────────────────┘
#
# Vitis AI Component Versions:
#   - unilog, xir, target-factory, vart: 3.5.0 (must match)
#   - xaiengine: 3.8 (from aie-rt repo)
#   - dynamic-dispatch: 1.2.0
#   - graph-engine, trace-logging: 1.0.0
#
# Supported Hardware:
#   - AMD Ryzen AI 300 Series (Strix Point) - PHX/HPT
#   - AMD Ryzen 8040 Series (Hawk Point)
#   - AMD Ryzen AI Max (Krackan Point)
#
# ============================================================================
{
  # ==========================================================================
  # XRT (Xilinx Runtime) and XDNA Driver
  # IMPORTANT: These two versions MUST match! Update both together.
  # ==========================================================================
  xrt = {
    version = "202610.2.21.21";  # Format: YYYYMM.MAJOR.MINOR.PATCH
    src = {
      owner = "Xilinx";
      repo = "XRT";
      rev = "202610.2.21.21";
      hash = "sha256-Foj33/U6waL81EzJ0ah66xCXEGWEkvhwmurKobfCevE=";
    };
  };

  xdna-driver = {
    version = "202610.2.21.21";  # MUST match xrt.version above!
    pluginVersion = "2.21.0";    # Shared library version (libxrt_driver_xdna.so.2.21.0)
    src = {
      owner = "amd";
      repo = "xdna-driver";
      rev = "202610.2.21.21";
      hash = "sha256-s06LKWwQNmWlmQSe+XNUOaVclnw1tAJPCFQvgDp/wCY=";
    };
  };

  # ==========================================================================
  # Vitis AI Components
  # Dependency chain: unilog → xir → target-factory → vart → graph-engine
  # All 3.5.0 components should be updated together for compatibility.
  # ==========================================================================
  unilog = {
    version = "3.5.0";  # Base logging library, no Vitis AI deps
    src = {
      owner = "amd";
      repo = "unilog";
      rev = "3abf8046d7ec8e651b8ec7ef19627a667ffaa741";
      hash = "sha256-TAsl/bCVwgVvbz3dQ9EKfBgZJArz4K2bae1hP/HuH3Q=";
    };
  };

  xir = {
    version = "3.5.0";
    src = {
      owner = "amd";
      repo = "xir";
      rev = "402341dd389d2f3ecd128e0a414d3fae3ca42db0";
      hash = "sha256-r6ZfIsUnUwVy31tb0JYWa24rxfnVl8W4AZbxqqVfc5s=";
    };
  };

  target-factory = {
    version = "3.5.0";
    src = {
      owner = "amd";
      repo = "target-factory";
      rev = "57bc90f72adcc92dd8f0ff8dbe393b69f4f30e8b";
      hash = "sha256-NR3GZJTYaFWwXLgPyBUmrL2sVoAPdwNBWsCqsKXpuUo=";
    };
  };

  vart = {
    version = "3.5.0";
    src = {
      owner = "amd";
      repo = "vart";
      rev = "9b5bd36afb2d2b3d3e88ac19b82a2d23a7b80d6e";
      hash = "sha256-U+n/RMY+wFUj2SkMW2pQCQUhQTSWVLGk9u0UH1jBfIY=";
    };
  };

  trace-logging = {
    version = "1.0.0";
    src = {
      owner = "amd";
      repo = "trace-logging";
      rev = "e23a668c1083ae3bcbe7a81de802fe51a88ff0fd";
      hash = "sha256-mvWADM2O0Ri2oEHvBVxSwTJDHNKV6nVAd8fHCcJw4GU=";
    };
  };

  graph-engine = {
    version = "1.0.0";
    src = {
      owner = "amd";
      repo = "graph_engine";
      rev = "d2c6f4db3f82eae6e2e1ebe3f5f0412f93bb22f2";
      hash = "sha256-T8p/rKzIy5OHdOWtL1w8wnDHpF3RCKP9RYXz8J1YVik=";
    };
  };

  xaiengine = {
    version = "3.8";
    src = {
      owner = "Xilinx";
      repo = "aie-rt";
      rev = "97eb0f58c2a07dfe1c51a79b53e9c39a8ef5db69";
      hash = "sha256-VDpFRm2gOVMqXDEEhQfqFQoEuJnB2CNK2P91cPwwqOc=";
    };
  };

  dynamic-dispatch = {
    version = "1.2.0";
    src = {
      owner = "amd";
      repo = "DynamicDispatch";
      rev = "bd17eec1fd8b60acfe6acb3f81f78b5aec6e8cd2";
      hash = "sha256-VeVQ2CKhIBJNE6ABt7BfNxNbr9VoLI8+1PQdG/vWPqo=";
    };
  };

  # MLIR-AIE / IRON
  mlir-aie = {
    version = "1.1.4";
    wheel = {
      url = "https://github.com/Xilinx/mlir-aie/releases/download/v1.1.4/mlir_aie-1.1.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
      hash = "sha256-kzFJxYjE6xxkefBXRTxjMeSsA2oJz9gvVn5L12OHPOI=";
    };
  };

  # AMD Proprietary Components (unfree)
  ryzen-ai-software = {
    version = "1.6.1";
    # Note: Requires manual download from AMD
    # https://www.amd.com/en/developer/resources/ryzen-ai-software.html
    filename = "ryzen-ai-rt-1.6.1-20250516.tar.gz";
    hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";  # Placeholder
  };

  # Applications
  whisper-iron = {
    version = "0.1.0";
    description = "Whisper speech recognition on AMD Ryzen AI NPU using IRON";
  };
}
