{ lib
, stdenv
, requireFile
, autoPatchelfHook
, python3
, zlib
, boost
, protobuf
, abseil-cpp
, xrt
}:

# AMD Ryzen AI Software - requires manual download from AMD Early Access Lounge
# https://www.amd.com/en/developer/resources/ryzen-ai-software.html
#
# This is an unfree package that repackages AMD's pre-built VitisAI EP libraries.
# Users must download ryzen_ai-1.6.1.tgz manually and add it to the Nix store.
#
# To use:
#   nix-store --add-fixed sha256 ryzen_ai-1.6.1.tgz
#   nix build .#ryzen-ai-software

stdenv.mkDerivation rec {
  pname = "ryzen-ai-software";
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
      4. Add to Nix store: nix-store --add-fixed sha256 ryzen_ai-${version}.tgz
    '';
  };

  nativeBuildInputs = [
    autoPatchelfHook
    python3
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    zlib
    boost
    protobuf
    abseil-cpp
    xrt
  ];

  unpackPhase = ''
    mkdir -p source
    tar -xzf $src -C source
  '';

  installPhase = ''
    mkdir -p $out/lib $out/share/xclbin $out/python

    # Install VitisAI EP shared library
    if [ -f source/venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime_vitisai_ep.so ]; then
      cp source/venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime_vitisai_ep.so $out/lib/
    fi

    # Install custom ops library
    if [ -d source/deployment/lib ]; then
      cp -r source/deployment/lib/* $out/lib/
    fi

    # Install xclbin files
    if [ -d source/xclbins ]; then
      cp -r source/xclbins/* $out/share/xclbin/
    fi

    # Install Python packages
    if [ -d source/venv/lib/python*/site-packages/voe ]; then
      cp -r source/venv/lib/python*/site-packages/voe $out/python/
    fi
  '';

  meta = with lib; {
    description = "AMD Ryzen AI Software - VitisAI EP runtime for NPU inference";
    homepage = "https://www.amd.com/en/developer/resources/ryzen-ai-software.html";
    license = licenses.unfree;  # AMD proprietary
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
