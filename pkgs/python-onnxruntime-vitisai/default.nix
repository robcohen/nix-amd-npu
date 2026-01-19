{ lib
, stdenv
, python3Packages
, onnxruntime-vitisai
, autoPatchelfHook
, xrt
}:

# Python bindings for ONNX Runtime with VitisAI EP
# This package uses the wheel from our VitisAI-enabled onnxruntime build
python3Packages.buildPythonPackage rec {
  pname = "onnxruntime-vitisai";
  inherit (onnxruntime-vitisai) version;
  format = "wheel";

  # Use the wheel from our VitisAI-enabled onnxruntime build
  src = "${onnxruntime-vitisai.dist}/onnxruntime-${version}-cp313-cp313-linux_x86_64.whl";

  nativeBuildInputs = lib.optionals stdenv.hostPlatform.isLinux [
    autoPatchelfHook
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    xrt
  ];

  # Runtime dependencies
  dependencies = with python3Packages; [
    coloredlogs
    numpy
    packaging
    # Note: flatbuffers, protobuf, sympy are bundled
  ];

  # Ensure VitisAI provider library is available
  postFixup = ''
    # Copy VitisAI provider if not already present
    if [ ! -f $out/lib/python*/site-packages/onnxruntime/capi/libonnxruntime_providers_vitisai.so ]; then
      echo "Copying VitisAI provider library..."
      cp ${onnxruntime-vitisai}/lib/libonnxruntime_providers_vitisai.so \
         $out/lib/python*/site-packages/onnxruntime/capi/ || true
    fi
  '';

  # Keep reference to C++ onnxruntime to prevent garbage collection
  passthru = {
    inherit onnxruntime-vitisai;
    tests = { };
  };

  pythonImportsCheck = [
    "onnxruntime"
  ];

  meta = with lib; {
    description = "ONNX Runtime Python bindings with VitisAI Execution Provider";
    homepage = "https://onnxruntime.ai";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
