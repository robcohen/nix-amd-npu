{ lib
, stdenv
, fetchFromGitHub
, cmake
, ninja
, pkg-config
, git
, python3
, boost
, rapidjson
, libdrm
, elfutils
, xrt
, linuxHeaders ? null
, kernel ? null
}:

stdenv.mkDerivation rec {
  pname = "xrt-plugin-amdxdna";
  version = "202610.2.21.21";
  pluginVersion = "2.21.0";

  src = fetchFromGitHub {
    owner = "amd";
    repo = "xdna-driver";
    rev = version;
    hash = "sha256-vXA8MzY0+KNquDG7jY3pZkm6lyM+V493xRmojl+wuIw=";
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
    rapidjson
    libdrm
    elfutils
    xrt
  ];

  cmakeDir = "../xrt/plugin/src";

  cmakeFlags = [
    "-DCMAKE_INSTALL_PREFIX=${placeholder "out"}/opt/xilinx"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DDISABLE_WERROR=ON"
    "-DXDNA_PLUGIN_VERSION_STRING=${pluginVersion}"
    # Point to XRT installation
    "-DXRT_INSTALL_PREFIX=${xrt}/opt/xilinx/xrt"
    # Disable kernel module (we use mainline)
    "-DXDNA_BUILD_DRIVER=OFF"
  ];

  postPatch = ''
    # Remove CPACK definitions that cause issues
    find . -name "CMakeLists.txt" -exec sed -i '/CPACK/d' {} \; || true

    # Remove kernel module build references
    find . -name "CMakeLists.txt" -exec sed -i '/driver\/amdxdna/d' {} \; || true

    # Disable Werror
    find . -name "CMakeLists.txt" -exec sed -i 's/-Werror//g' {} \; || true
  '';

  postInstall = ''
    # Move plugin library to XRT's expected location
    mkdir -p $out/opt/xilinx/xrt/lib
    if [ -d $out/opt/xilinx/lib ]; then
      mv $out/opt/xilinx/lib/* $out/opt/xilinx/xrt/lib/ || true
    fi

    # Create top-level symlinks
    mkdir -p $out/lib
    for lib in $out/opt/xilinx/xrt/lib/*.so*; do
      ln -sf $lib $out/lib/ 2>/dev/null || true
    done

    # Create pkg-config file
    mkdir -p $out/lib/pkgconfig
    cat > $out/lib/pkgconfig/xrt-amdxdna.pc << EOF
    prefix=$out/opt/xilinx/xrt
    exec_prefix=\''${prefix}
    libdir=\''${exec_prefix}/lib
    includedir=\''${prefix}/include

    Name: XRT-AMDXDNA
    Description: AMD XDNA Plugin for Xilinx Runtime
    Version: ${pluginVersion}
    Requires: xrt
    Libs: -L\''${libdir} -lamdxdna
    EOF
  '';

  meta = with lib; {
    description = "AMD XDNA driver plugin for XRT (Ryzen AI NPU support)";
    homepage = "https://github.com/amd/xdna-driver";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
