{ inputs, ... }:
{
  perSystem = { config, pkgs, ... }:
    let
      xrt = config.packages.xrt;
      xrt-amdxdna = config.packages.xrt-amdxdna;
      onnxruntime-vitisai = config.packages.onnxruntime-vitisai;
      mlir-aie = config.packages.mlir-aie;

      # Shared NPU detection script for all shells
      npuDetectionScript = ''
        # Check for NPU hardware
        _check_npu() {
          if [ -e /dev/accel/accel0 ]; then
            echo -e "\033[32m[OK]\033[0m NPU device found at /dev/accel/accel0"
            if [ -r /dev/accel/accel0 ]; then
              echo -e "\033[32m[OK]\033[0m NPU device is readable"
            else
              echo -e "\033[33m[WARN]\033[0m NPU device not readable - check group membership"
              echo "       Add yourself to the 'video' group: sudo usermod -aG video $USER"
            fi
          else
            echo -e "\033[33m[WARN]\033[0m NPU device not found at /dev/accel/accel0"
            echo "       - Ensure kernel 6.10+ is running (current: $(uname -r))"
            echo "       - Check if amdxdna module is loaded: lsmod | grep amdxdna"
            echo "       - Verify IOMMU is enabled: dmesg | grep -i iommu"
          fi
        }
      '';
    in
    {
      devShells = {
        # Basic XRT development shell
        default = pkgs.mkShell {
          packages = [
            xrt-amdxdna
          ];

          shellHook = ''
            ${npuDetectionScript}

            echo "=============================================="
            echo "  AMD Ryzen AI NPU Development Environment"
            echo "=============================================="
            echo "XRT version: ${xrt.version}"
            echo ""
            export XILINX_XRT="${xrt-amdxdna}/opt/xilinx/xrt"
            export LD_LIBRARY_PATH="${xrt-amdxdna}/opt/xilinx/xrt/lib:''${LD_LIBRARY_PATH:-}"

            echo "NPU Status:"
            _check_npu
            echo ""
            echo "Commands:"
            echo "  xrt-smi examine    - Show NPU hardware details"
            echo "  xrt-smi validate   - Run NPU validation tests"
            echo ""
          '';
        };

        # VitisAI development shell with ONNX Runtime (from-source only)
        vitisai = pkgs.mkShell {
          packages = [
            xrt-amdxdna
            onnxruntime-vitisai
            config.packages.python-onnxruntime-vitisai
            config.packages.dynamic-dispatch
            (pkgs.python313.withPackages (ps: [
              ps.numpy
              ps.coloredlogs
              ps.packaging
            ]))
          ];

          shellHook = ''
            ${npuDetectionScript}

            echo "=============================================="
            echo "  AMD Ryzen AI NPU + ONNX Runtime VitisAI EP"
            echo "=============================================="
            echo "XRT version: ${xrt.version}"
            echo "ONNX Runtime: ${onnxruntime-vitisai.version}"
            echo ""

            export XILINX_XRT="${xrt-amdxdna}/opt/xilinx/xrt"
            export LD_LIBRARY_PATH="${xrt-amdxdna}/opt/xilinx/xrt/lib:${onnxruntime-vitisai}/lib:''${LD_LIBRARY_PATH:-}"
            export PYTHONPATH="${config.packages.python-onnxruntime-vitisai}/lib/python3.13/site-packages:''${PYTHONPATH:-}"

            echo "NPU Status:"
            _check_npu
            echo ""
            echo "VitisAI EP: REGISTERED"
            echo ""
            echo "Test with:"
            echo "  python -c 'import onnxruntime as ort; print(ort.get_available_providers())'"
            echo ""
            echo "NOTE: For full NPU inference, you need AMD's VAIP runtime from:"
            echo "      https://www.amd.com/en/developer/resources/ryzen-ai-software.html"
            echo ""
          '';
        };

        # Note: ryzen-ai-full shell requires unfree packages
        # Use: NIXPKGS_ALLOW_UNFREE=1 nix develop .#ryzen-ai-full --impure

        # MLIR-AIE / IRON development shell for custom NPU kernels
        iron = pkgs.mkShell {
          packages = [
            xrt-amdxdna
            mlir-aie
            (pkgs.python312.withPackages (ps: with ps; [
              numpy
              scipy
              pytest
              # Audio/ML deps
              librosa
              soundfile
              transformers
              torch
              # For MLIR-AIE
              pybind11
              ml-dtypes
            ]))
            pkgs.cmake
            pkgs.ninja
            pkgs.clang
            pkgs.lld
            pkgs.xorg.libXrender
            pkgs.xorg.libXtst
            pkgs.xorg.libXi
            # Audio processing for Whisper
            pkgs.ffmpeg
            pkgs.libsndfile
          ];

          shellHook = ''
            ${npuDetectionScript}

            echo "=============================================="
            echo "  AMD Ryzen AI NPU - MLIR-AIE / IRON Dev"
            echo "=============================================="
            echo "XRT version: ${xrt.version}"
            echo "MLIR-AIE version: ${mlir-aie.version}"
            echo ""

            export XILINX_XRT="${xrt-amdxdna}/opt/xilinx/xrt"
            export LD_LIBRARY_PATH="${xrt-amdxdna}/opt/xilinx/xrt/lib:''${LD_LIBRARY_PATH:-}"

            # Add mlir-aie site-packages (for aie.pth to work)
            export PYTHONPATH="${mlir-aie}/lib/python3.12/site-packages:${mlir-aie}/lib/python3.12/site-packages/mlir_aie/python:$PWD/pkgs/whisper-iron:''${PYTHONPATH:-}"

            # Add mlir-aie binaries to PATH
            export PATH="${mlir-aie}/lib/python3.12/site-packages/mlir_aie/bin:''${PATH:-}"

            echo "NPU Status:"
            _check_npu
            echo ""
            echo "Whisper-IRON: Speech recognition on AMD NPU"
            echo ""
            echo "Quick start:"
            echo "  cd pkgs/whisper-iron"
            echo "  python tests/test_npu.py        # Run NPU tests"
            echo "  python transcribe.py audio.wav  # Transcribe audio"
            echo ""
          '';
        };

        # Full IRON development shell using FHS environment for pip compatibility
        # This allows installing eudsl-python-extras from source
        iron-full =
          let
            # Use XDG_CACHE_HOME for venv to avoid polluting project directory
            venvDir = "\${XDG_CACHE_HOME:-$HOME/.cache}/nix-amd-npu/venv-iron";
            fhsEnv = pkgs.buildFHSEnv {
              name = "iron-fhs";
              targetPkgs = pkgs: [
                xrt-amdxdna
                pkgs.python312
                pkgs.python312Packages.pip
                pkgs.python312Packages.virtualenv
                # Build dependencies for eudsl-python-extras
                pkgs.cmake
                pkgs.ninja
                pkgs.clang_18
                pkgs.lld_18
                pkgs.git
                # Runtime dependencies
                pkgs.zlib
                pkgs.ncurses
                pkgs.libxml2
                pkgs.stdenv.cc.cc.lib
                # For audio processing
                pkgs.ffmpeg
                pkgs.libsndfile
              ];
              runScript = "bash";
              profile = ''
                export XILINX_XRT="${xrt-amdxdna}/opt/xilinx/xrt"
                export LD_LIBRARY_PATH="${xrt-amdxdna}/opt/xilinx/xrt/lib:''${LD_LIBRARY_PATH:-}"

                # Create virtualenv in cache directory if it doesn't exist
                VENV_DIR="${venvDir}"
                mkdir -p "$(dirname "$VENV_DIR")"
                if [ ! -d "$VENV_DIR" ]; then
                  echo "Creating Python virtual environment at $VENV_DIR..."
                  python3 -m venv "$VENV_DIR"
                fi
                source "$VENV_DIR/bin/activate"
                echo "Activated virtualenv: $VENV_DIR"
              '';
            };
          in
          pkgs.mkShell {
            packages = [ fhsEnv ];
            shellHook = ''
              ${npuDetectionScript}

              echo "=============================================="
              echo "  AMD Ryzen AI NPU - Full IRON Dev (FHS)"
              echo "=============================================="
              echo ""
              echo "NPU Status:"
              _check_npu
              echo ""
              echo "Run 'iron-fhs' to enter the FHS environment, then:"
              echo ""
              echo "Setup IRON (first time only):"
              echo "  pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-2"
              echo "  pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"
              echo "  EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX=aie pip install eudsl-python-extras -f https://llvm.github.io/eudsl"
              echo ""
              echo "Then test:"
              echo "  python -c 'from aie.iron import ObjectFifo; print(\"IRON works!\")'"
              echo ""
            '';
          };

        # Whisper-IRON shell (same as iron, kept for backwards compat)
        whisper = pkgs.mkShell {
          packages = [
            xrt-amdxdna
            mlir-aie
            (pkgs.python312.withPackages (ps: with ps; [
              numpy
              scipy
              pytest
              librosa
              soundfile
              transformers
              torch
              # For MLIR-AIE
              ml-dtypes
            ]))
            pkgs.cmake
            pkgs.ninja
            pkgs.clang
            pkgs.lld
            pkgs.xorg.libXrender
            pkgs.xorg.libXtst
            pkgs.xorg.libXi
            pkgs.ffmpeg
            pkgs.libsndfile
          ];

          shellHook = ''
            ${npuDetectionScript}

            echo "=============================================="
            echo "  Whisper-IRON: Speech Recognition on NPU"
            echo "=============================================="
            echo "XRT version: ${xrt.version}"
            echo "MLIR-AIE version: ${mlir-aie.version}"
            echo ""

            export XILINX_XRT="${xrt-amdxdna}/opt/xilinx/xrt"
            export LD_LIBRARY_PATH="${xrt-amdxdna}/opt/xilinx/xrt/lib:''${LD_LIBRARY_PATH:-}"
            export PYTHONPATH="${mlir-aie}/lib/python3.12/site-packages:${mlir-aie}/lib/python3.12/site-packages/mlir_aie/python:$PWD/pkgs/whisper-iron:''${PYTHONPATH:-}"
            export PATH="${mlir-aie}/lib/python3.12/site-packages/mlir_aie/bin:''${PATH:-}"

            echo "NPU Status:"
            _check_npu
            echo ""
            echo "Usage:"
            echo "  cd pkgs/whisper-iron"
            echo "  python tests/test_npu.py        # Test kernels"
            echo "  python transcribe.py audio.wav  # Transcribe"
            echo ""
          '';
        };
      };
    };
}
