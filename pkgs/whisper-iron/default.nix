# Whisper-IRON: Speech recognition on AMD Ryzen AI NPU
#
# This package provides a Whisper implementation that can run on AMD NPUs
# using the IRON (Intermediate Representation for Open Neural networks) framework.
#
{ lib
, python312Packages
, mlir-aie
, xrt-amdxdna
, ffmpeg
, libsndfile
}:

python312Packages.buildPythonApplication rec {
  pname = "whisper-iron";
  version = "0.1.0";
  pyproject = false;

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./kernels
      ./model
      ./utils
      ./tests
      ./transcribe.py
      ./requirements.txt
    ];
  };

  propagatedBuildInputs = with python312Packages; [
    numpy
    scipy
    librosa
    soundfile
    transformers
    torch
    ml-dtypes
  ];

  nativeCheckInputs = with python312Packages; [
    pytest
  ];

  # Python version for path construction
  pythonVersion = "python3.12";
  sitePackages = "$out/lib/${pythonVersion}/site-packages";

  # Install as a Python package
  installPhase = ''
    runHook preInstall

    mkdir -p ${sitePackages}/whisper_iron
    cp -r kernels model utils ${sitePackages}/whisper_iron/

    # Create __init__.py
    cat > ${sitePackages}/whisper_iron/__init__.py << 'EOF'
"""Whisper-IRON: Speech recognition on AMD Ryzen AI NPU"""
from .model.whisper import WhisperModel
from .model.config import WhisperConfig, WHISPER_CONFIGS
__version__ = "${version}"
__all__ = ["WhisperModel", "WhisperConfig", "WHISPER_CONFIGS"]
EOF

    # Install CLI script
    mkdir -p $out/bin
    cat > $out/bin/whisper-iron << SCRIPT
#!/usr/bin/env python3
import sys
sys.path.insert(0, "${sitePackages}")
exec(open("${sitePackages}/whisper_iron/transcribe.py").read())
SCRIPT
    chmod +x $out/bin/whisper-iron

    # Copy transcribe.py
    cp transcribe.py ${sitePackages}/whisper_iron/

    runHook postInstall
  '';

  # Skip tests that require NPU hardware
  doCheck = false;

  passthru = {
    # Allow running tests manually with: nix build .#whisper-iron.tests
    tests = python312Packages.runCommand "whisper-iron-tests" {
      nativeBuildInputs = [ python312Packages.pytest ];
    } ''
      cd ${src}
      pytest tests/test_kernels.py -v
      touch $out
    '';
  };

  meta = with lib; {
    description = "Whisper speech recognition on AMD Ryzen AI NPU using IRON";
    longDescription = ''
      Whisper-IRON is a speech recognition implementation that runs on AMD Ryzen AI NPUs.
      It uses the IRON (Intermediate Representation for Open Neural networks) framework
      from MLIR-AIE to compile and execute neural network kernels on the NPU.

      Supports Whisper tiny, base, and small models with automatic CPU fallback
      when NPU acceleration is not available.
    '';
    homepage = "https://github.com/nix-amd-npu/nix-amd-npu";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
    mainProgram = "whisper-iron";
  };
}
