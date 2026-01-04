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
, ncurses
, gawk
, libsystemtap
, wget
, cacert
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
    wget
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
    ncurses
    libsystemtap
  ];

  cmakeDir = "../src";

  cmakeFlags = [
    "-DCMAKE_INSTALL_PREFIX=${placeholder "out"}/opt/xilinx/xrt"
    "-DXRT_INSTALL_PREFIX=${placeholder "out"}/opt/xilinx/xrt"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DDISABLE_WERROR=ON"
    # Disable kernel module building (we use mainline amdxdna)
    "-DXRT_DKMS_DRIVER_SRC_BASE_DIR="
    # Disable static linking for aiebu tools
    "-DAIEBU_STATIC_BUILD=OFF"
  ];

  # Skip building kernel modules and fix Nix-specific issues
  postPatch = ''
    # Remove kernel module references
    substituteInPlace src/CMakeLists.txt \
      --replace-quiet 'add_subdirectory(runtime_src/core/pcie/driver)' '#add_subdirectory(runtime_src/core/pcie/driver)' || true

    # Fix hardcoded paths
    substituteInPlace src/runtime_src/core/common/config_reader.cpp \
      --replace-quiet '/opt/xilinx/xrt' '${placeholder "out"}/opt/xilinx/xrt' || true

    # Fix /etc/os-release access - create a fake one for the build
    mkdir -p $TMPDIR/etc
    cat > $TMPDIR/etc/os-release << EOF2
    ID=nixos
    VERSION_ID="25.11"
    EOF2

    # Patch CMake scripts that try to read /etc/os-release
    find . -name "*.cmake" -o -name "CMakeLists.txt" | xargs sed -i \
      -e 's|/etc/os-release|'$TMPDIR'/etc/os-release|g' || true

    # Disable Werror
    find . -name "CMakeLists.txt" -exec sed -i 's/-Werror//g' {} \; || true

    # Remove static linking flags that cause issues on Nix
    find . -name "CMakeLists.txt" -exec sed -i 's/-static//g' {} \; || true
    find . -name "*.cmake" -exec sed -i 's/-static//g' {} \; || true

    # Also remove STATIC_LINK variable usage and set_target_properties LINK_FLAGS -static
    find . -name "CMakeLists.txt" -exec sed -i 's/STATIC_LINK/-DSTATIC_DISABLED/g' {} \; || true
    find . -name "CMakeLists.txt" -exec sed -i 's/set_target_properties.*LINK_FLAGS.*-static.*)/# Disabled for Nix/g' {} \; || true

    # Specifically patch the aiebu static linking
    if [ -d src/runtime_src/core/common/aiebu ]; then
      find src/runtime_src/core/common/aiebu -name "CMakeLists.txt" -exec sed -i \
        -e 's/LINK_FLAGS "-static"/LINK_FLAGS ""/g' \
        -e 's/"-static"/""/g' \
        {} \;
    fi

    # Disable the dynamic dependency checker (fails because we use dynamic linking)
    if [ -f src/runtime_src/core/common/aiebu/cmake/depends.cmake ]; then
      echo "# Disabled for Nix build" > src/runtime_src/core/common/aiebu/cmake/depends.cmake
    fi

    # Create the markdown_graphviz_svg.py file to avoid network fetch during install
    mkdir -p src/runtime_src/core/common/aiebu/specification
    cat > src/runtime_src/core/common/aiebu/specification/markdown_graphviz_svg.py << 'PYEOF'
# Stub file - actual content not needed for runtime
pass
PYEOF

    # Patch out wget download during install
    find . -name "CMakeLists.txt" -exec sed -i 's/wget.*markdown_graphviz_svg.py/true/g' {} \; || true
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
