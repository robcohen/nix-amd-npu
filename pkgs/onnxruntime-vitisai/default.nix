{ lib
, onnxruntime
, xrt
}:

# Extend nixpkgs onnxruntime with VitisAI EP support
onnxruntime.overrideAttrs (oldAttrs: {
  pname = "onnxruntime-vitisai";

  cmakeFlags = oldAttrs.cmakeFlags ++ [
    "-Donnxruntime_USE_VITISAI=ON"
    "-Donnxruntime_USE_FULL_PROTOBUF=ON"
    "-DXRT_INCLUDE_DIRS=${xrt}/opt/xilinx/xrt/include"
  ];

  buildInputs = oldAttrs.buildInputs ++ [ xrt ];

  # Fix GCC 15 compatibility and missing defines in VitisAI provider
  postPatch = (oldAttrs.postPatch or "") + ''
    # Add missing cstdint includes for GCC 15
    for f in onnxruntime/core/providers/vitisai/imp/*.cc \
             onnxruntime/core/providers/vitisai/imp/*.h; do
      if [ -f "$f" ]; then
        if ! grep -q '#include <cstdint>' "$f"; then
          sed -i '1i #include <cstdint>' "$f"
        fi
      fi
    done

    # Define GIT_COMMIT_ID if not set
    sed -i 's/GIT_COMMIT_ID/"unknown"/g' onnxruntime/core/providers/vitisai/imp/global_api.cc
  '';

  # VitisAI EP loads onnxruntime_vitisai_ep.so dynamically
  # This library must be provided by AMD's VAIP (built separately)
  passthru = (oldAttrs.passthru or {}) // {
    vitisaiEPInfo = ''
      This package includes the ONNX Runtime VitisAI execution provider wrapper.

      At runtime, it dynamically loads 'onnxruntime_vitisai_ep.so' which must be
      provided separately (from AMD's VAIP build or Ryzen AI Software package).

      Set LD_LIBRARY_PATH to include the directory containing onnxruntime_vitisai_ep.so
    '';
  };

  meta = oldAttrs.meta // {
    description = "ONNX Runtime with VitisAI Execution Provider for AMD Ryzen AI NPU";
  };
})
