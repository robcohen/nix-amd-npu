{ lib
, stdenv
, fetchFromGitHub
, cmake
, ninja
, pkg-config
, git
, python3
, boost
, opencl-headers
, opencl-clhpp
, ocl-icd
, rapidjson
, protobuf
, elfutils
, libdrm
, systemd
, curl
, openssl
, libuuid
, libxcrypt
, pybind11
}:

stdenv.mkDerivation rec {
  pname = "xrt";
  version = "202610.2.21.21";

  src = fetchFromGitHub {
    owner = "Xilinx";
    repo = "XRT";
    rev = version;
    hash = "sha256-Foj33/U6waL81EzJ0ah66xCXEGWEkvhwmurKobfCevE=";
    fetchSubmodules = true;
  };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    git
    python3
  ];

  buildInputs = [
    boost
    opencl-headers
    opencl-clhpp
    ocl-icd
    rapidjson
    protobuf
    elfutils
    libdrm
    systemd
    curl
    openssl
    libuuid
    libxcrypt
    pybind11
  ];

  cmakeDir = "../src";

  cmakeFlags = [
    "-DCMAKE_INSTALL_PREFIX=${placeholder "out"}/opt/xilinx/xrt"
    "-DXRT_INSTALL_PREFIX=${placeholder "out"}/opt/xilinx/xrt"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DDISABLE_WERROR=ON"
    # Disable kernel module building (we use mainline amdxdna)
    "-DXRT_DKMS_DRIVER_SRC_BASE_DIR="
  ];

  # Skip building kernel modules
  postPatch = ''
    # Remove kernel module references
    substituteInPlace src/CMakeLists.txt \
      --replace-quiet 'add_subdirectory(runtime_src/core/pcie/driver)' '#add_subdirectory(runtime_src/core/pcie/driver)' || true

    # Fix hardcoded paths
    substituteInPlace src/runtime_src/core/common/config_reader.cpp \
      --replace-quiet '/opt/xilinx/xrt' '${placeholder "out"}/opt/xilinx/xrt' || true
  '';

  postInstall = ''
    # Create convenience symlinks at top level
    mkdir -p $out/bin $out/lib $out/include

    # Link binaries
    for bin in $out/opt/xilinx/xrt/bin/*; do
      ln -sf $bin $out/bin/
    done

    # Link libraries
    for lib in $out/opt/xilinx/xrt/lib/*.so*; do
      ln -sf $lib $out/lib/
    done

    # Copy setup script
    cp $out/opt/xilinx/xrt/setup.sh $out/ || true

    # Create pkg-config file
    mkdir -p $out/lib/pkgconfig
    cat > $out/lib/pkgconfig/xrt.pc << EOF
    prefix=$out/opt/xilinx/xrt
    exec_prefix=\''${prefix}
    libdir=\''${exec_prefix}/lib
    includedir=\''${prefix}/include

    Name: XRT
    Description: Xilinx Runtime for AMD NPU
    Version: ${version}
    Libs: -L\''${libdir} -lxrt_coreutil
    Cflags: -I\''${includedir}
    EOF
  '';

  meta = with lib; {
    description = "Xilinx Runtime (XRT) for AMD Ryzen AI NPU";
    homepage = "https://github.com/Xilinx/XRT";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
