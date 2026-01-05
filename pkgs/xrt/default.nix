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

  # Python with packages needed by spec_tool.py during build
  pythonWithPackages = python3.withPackages (ps: [
    ps.pyyaml
    ps.markdown
    ps.jinja2
    ps.pybind11
  ]);

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    git
    pythonWithPackages
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
    # XRT_UPSTREAM_DEBIAN enables XRT_UPSTREAM which propagates to AIEBU_UPSTREAM
    # This disables static linking in aiebu tools
    # See: src/CMake/settings.cmake lines 19-21
    "-DXRT_UPSTREAM_DEBIAN=ON"
    # Override install dirs to relative paths to prevent aiebu cmake path issues
    # aiebu concatenates CMAKE_BINARY_DIR + CMAKE_INSTALL_LIBDIR which breaks with absolute paths
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DCMAKE_INSTALL_BINDIR=bin"
    "-DCMAKE_INSTALL_INCLUDEDIR=include"
  ];

  # Skip building kernel modules and fix Nix-specific issues
  postPatch = ''
    # Remove kernel module references
    substituteInPlace src/CMakeLists.txt \
      --replace-quiet 'add_subdirectory(runtime_src/core/pcie/driver)' '#add_subdirectory(runtime_src/core/pcie/driver)' || true

    # Fix hardcoded /usr/src DKMS install path
    # Redirect to $out/share/xrt-dkms-src instead of /usr/src
    for f in src/CMake/version.cmake src/CMake/dkms.cmake src/CMake/dkms-edge.cmake; do
      if [ -f "$f" ]; then
        sed -i 's|/usr/src/xrt-|''${CMAKE_INSTALL_PREFIX}/share/xrt-dkms-src/xrt-|g' "$f"
      fi
    done
    if [ -f src/CMake/dkms-aws.cmake ]; then
      sed -i 's|/usr/src/xrt-aws-|''${CMAKE_INSTALL_PREFIX}/share/xrt-dkms-src/xrt-aws-|g' src/CMake/dkms-aws.cmake
    fi

    # Fix hardcoded /usr/local/bin install paths for xbflash tools
    for f in src/runtime_src/core/tools/xbflash2/CMakeLists.txt src/runtime_src/core/pcie/tools/xbflash.qspi/CMakeLists.txt; do
      if [ -f "$f" ]; then
        sed -i 's|"/usr/local/bin"|"''${CMAKE_INSTALL_PREFIX}/bin"|g' "$f"
      fi
    done

    # Fix /etc/OpenCL/vendors path for OpenCL ICD registration
    if [ -f src/CMake/icd.cmake ]; then
      sed -i 's|/etc/OpenCL/vendors|''${CMAKE_INSTALL_PREFIX}/etc/OpenCL/vendors|g' src/CMake/icd.cmake
    fi

    # Fix hardcoded paths
    substituteInPlace src/runtime_src/core/common/config_reader.cpp \
      --replace-quiet '/opt/xilinx/xrt' '${placeholder "out"}/opt/xilinx/xrt' || true

    # Fix /etc/os-release access - create a fake one for the build
    mkdir -p $TMPDIR/etc
    echo 'ID=nixos' > $TMPDIR/etc/os-release
    echo 'VERSION_ID="25.11"' >> $TMPDIR/etc/os-release

    # Patch CMake scripts that try to read /etc/os-release
    find . -name "*.cmake" -o -name "CMakeLists.txt" | xargs sed -i \
      -e 's|/etc/os-release|'$TMPDIR'/etc/os-release|g' || true

    # Disable Werror globally
    find . -name "CMakeLists.txt" -exec sed -i 's/-Werror//g' {} \; || true

    # Note: XRT_UPSTREAM_DEBIAN=ON sets AIEBU_UPSTREAM which disables static linking
    # via "if (NOT AIEBU_UPSTREAM)" guards in the CMake files

    # Create stub markdown_graphviz_svg.py module to avoid network download
    # The spec_tool.py imports this as a markdown extension
    # The spec_tool uses GraphvizBlocksExtension() for HTML doc generation
    cat > src/runtime_src/core/common/aiebu/specification/markdown_graphviz_svg.py << 'PYEOF'
# Stub implementation of markdown_graphviz_svg for Nix build
# The real module is https://github.com/Tanami/markdown-graphviz-svg
# This stub provides minimal interface to avoid import errors

from markdown.extensions import Extension

class GraphvizBlocksExtension(Extension):
    """Stub extension - graphviz rendering is disabled in Nix build"""
    def extendMarkdown(self, md):
        pass

# Also provide the original class names for compatibility
GraphvizExtension = GraphvizBlocksExtension

def makeExtension(**kwargs):
    return GraphvizBlocksExtension(**kwargs)
PYEOF

    # Replace wget command in specification CMakeLists.txt
    # The wget downloads a Python module - we use our stub instead
    # The command is split across multiple lines, so just replace 'wget' with 'true #'
    find . -name "CMakeLists.txt" -exec grep -l "wget" {} \; | while read f; do
      echo "Patching wget out of: $f"
      # Replace wget with true - this disables the network download
      # The stub file is pre-created in preInstall phase
      sed -i 's|COMMAND wget|COMMAND true # wget|g' "$f"
      sed -i 's|COMMAND powershell wget|COMMAND true # powershell wget|g' "$f"
    done

    # Disable the spec generation targets during install
    # The build phase generates these files correctly, but install-time regeneration
    # with spec_tool.py produces incomplete output (no disassembler code)
    # Replace entire add_custom_target blocks with empty targets
    specCmake="src/runtime_src/core/common/aiebu/specification/aie2ps/CMakeLists.txt"
    if [ -f "$specCmake" ]; then
      echo "Disabling spec generation in $specCmake"
      # Create empty stub CMakeLists that does nothing
      cat > "$specCmake" << 'STUBCMAKE'
# SPDX-License-Identifier: MIT
# Disabled for Nix build - spec generation causes issues
# The ISA headers are generated during build phase
message(STATUS "Skipping aie2ps spec generation (Nix build)")
STUBCMAKE
    fi

    # Fix spec_tool.py shebang issue - the script uses #!/usr/bin/env python3
    # which doesn't work in Nix sandbox (no /usr/bin/env)
    # Use patchShebangs to fix all Python scripts to use the Nix Python
    patchShebangs --build src/runtime_src/core/common/aiebu/specification/
    patchShebangs --build src/runtime_src/core/common/aiebu/src/python/ || true

    # Verify the stub exists where aiebu expects it
    echo "Created stub at: src/runtime_src/core/common/aiebu/specification/markdown_graphviz_svg.py"
    ls -la src/runtime_src/core/common/aiebu/specification/markdown_graphviz_svg.py
  '';

  # Note: markdown_graphviz_svg.py stub is created once in postPatch.
  # The preInstall stub was removed as spec generation is disabled via CMakeLists.txt stub.

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
    # For nixpkgs: maintainers = with maintainers; [ robcohen ];
    maintainers = [ ];
  };
}
