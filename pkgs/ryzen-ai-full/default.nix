{ lib
, stdenv
, makeWrapper
, symlinkJoin
, requireFile
, autoPatchelfHook
, python3
, zlib
, boost
, protobuf
, abseil-cpp
# Our from-source packages
, xrt
, xrt-amdxdna
, onnxruntime-vitisai
, dynamic-dispatch
, unilog
, xir
, vart
, graph-engine
}:

let
  # AMD's pre-built VAIP runtime (requires manual download)
  vaip-runtime = stdenv.mkDerivation rec {
    pname = "vaip-runtime";
    version = "1.6.1";

    src = requireFile {
      name = "ryzen_ai-${version}.tgz";
      url = "https://www.amd.com/en/developer/resources/ryzen-ai-software.html";
      hash = "sha256-0000000000000000000000000000000000000000000=";  # Update after download
      message = ''
        AMD Ryzen AI Software requires registration to download.

        1. Visit: https://www.amd.com/en/developer/resources/ryzen-ai-software.html
        2. Register for AMD Early Access Lounge
        3. Download ryzen_ai-${version}.tgz
        4. Add to Nix store:
           nix-store --add-fixed sha256 ryzen_ai-${version}.tgz
      '';
    };

    nativeBuildInputs = [
      autoPatchelfHook
    ];

    buildInputs = [
      stdenv.cc.cc.lib
      zlib
      boost
      protobuf
      abseil-cpp
      xrt
      python3
    ];

    dontBuild = true;

    unpackPhase = ''
      mkdir -p source
      tar -xzf $src -C source
    '';

    installPhase = ''
      mkdir -p $out/lib $out/share/xclbin $out/share/vaip

      # Find and install VitisAI EP library
      find source -name "libonnxruntime_vitisai_ep.so*" -exec cp {} $out/lib/ \;
      find source -name "libvoe*.so*" -exec cp {} $out/lib/ \;
      find source -name "libonnx_custom_ops.so*" -exec cp {} $out/lib/ \;

      # Install xclbin files
      find source -name "*.xclbin" -exec cp {} $out/share/xclbin/ \;

      # Install config files
      find source -name "vaip_config.json" -exec cp {} $out/share/vaip/ \;
    '';

    meta = with lib; {
      description = "AMD VAIP runtime for Ryzen AI NPU";
      license = licenses.unfree;
      platforms = [ "x86_64-linux" ];
    };
  };

in symlinkJoin {
  name = "ryzen-ai-full-1.6.1";

  paths = [
    xrt-amdxdna
    onnxruntime-vitisai
    dynamic-dispatch
    vaip-runtime
  ];

  nativeBuildInputs = [ makeWrapper ];

  postBuild = ''
    # Create environment setup script
    mkdir -p $out/etc/profile.d
    cat > $out/etc/profile.d/ryzen-ai.sh << 'ENVSCRIPT'
#!/bin/bash
# AMD Ryzen AI NPU Environment Setup

export XILINX_XRT="@xrt@/opt/xilinx/xrt"
export LD_LIBRARY_PATH="@out@/lib:@xrt@/opt/xilinx/xrt/lib:''${LD_LIBRARY_PATH:-}"
export XLNX_VART_FIRMWARE="@out@/share/xclbin"

# VitisAI EP configuration
export VAIP_CONFIG="@out@/share/vaip/vaip_config.json"

echo "Ryzen AI NPU environment configured"
echo "  XILINX_XRT=$XILINX_XRT"
echo "  XLNX_VART_FIRMWARE=$XLNX_VART_FIRMWARE"
ENVSCRIPT

    substituteInPlace $out/etc/profile.d/ryzen-ai.sh \
      --replace @out@ $out \
      --replace @xrt@ ${xrt-amdxdna}

    # Create wrapper for xrt-smi
    if [ -x $out/bin/unwrapped/xrt-smi ]; then
      wrapProgram $out/bin/unwrapped/xrt-smi \
        --set XILINX_XRT "${xrt-amdxdna}/opt/xilinx/xrt"
    fi
  '';

  passthru = {
    inherit vaip-runtime xrt-amdxdna onnxruntime-vitisai;

    # Export environment for shell
    shellHook = ''
      source ${placeholder "out"}/etc/profile.d/ryzen-ai.sh
    '';
  };

  meta = with lib; {
    description = "Complete AMD Ryzen AI NPU stack for NixOS";
    homepage = "https://www.amd.com/en/developer/resources/ryzen-ai-software.html";
    license = licenses.unfree;  # Contains AMD proprietary components
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
    longDescription = ''
      Complete AMD Ryzen AI NPU stack combining:
      - XRT (built from source) - Xilinx Runtime
      - XDNA driver plugin (built from source)
      - ONNX Runtime with VitisAI EP (built from source)
      - Vitis AI libraries (built from source)
      - AMD VAIP runtime (pre-built, requires manual download)
      - xclbin files for NPU

      After installing, source the environment:
        source /etc/profile.d/ryzen-ai.sh

      Or use the devShell:
        nix develop .#ryzen-ai-full
    '';
  };
}
