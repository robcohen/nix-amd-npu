{ lib
, stdenv
, fetchFromGitHub
, cmake
, ninja
, pkg-config
, git
, boost
, libuuid
, unilog
, xir
, vart
, xrt
}:

stdenv.mkDerivation rec {
  pname = "graph-engine";
  version = "1.0.0";

  src = fetchFromGitHub {
    owner = "amd";
    repo = "graph_engine";
    rev = "2410822ec0450ff3602ef26bbe074685765c9144";
    hash = "sha256-W9vtQ/wq6l0gT2B5w51NBK0AB6MG3dspZSzTqCzUMgU=";
  };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    git
  ];

  buildInputs = [
    boost
    libuuid
    unilog
    xir
    vart
    xrt
  ];

  cmakeFlags = [
    "-DBUILD_TEST=OFF"
    "-DBUILD_SHARED_LIBS=ON"
    # Point to XRT cmake config
    "-DXRT_DIR=${xrt}/opt/xilinx/xrt/share/cmake/XRT"
    "-DCMAKE_PREFIX_PATH=${xrt}/opt/xilinx/xrt"
  ];

  postPatch = ''
    # Keep AVX-512 flags - required for SIMD operations in graph-engine
    # This limits compatibility to CPUs with AVX-512 support
    # Note: AMD Zen 4+ and Intel Ice Lake+ support these
    :

    # Create a fake git repo for version detection
    git init
    git config user.email "nix@build"
    git config user.name "Nix Build"
    git add -A
    git commit -m "Nix build" --allow-empty

    # Fix GCC 15 compatibility - add missing includes
    for f in $(find . -name "*.cpp" -o -name "*.hpp"); do
      if grep -q 'uint64_t\|int64_t\|uint32_t\|int32_t' "$f" 2>/dev/null; then
        if ! grep -q '#include <cstdint>' "$f" 2>/dev/null; then
          sed -i '1i #include <cstdint>' "$f"
        fi
      fi
    done
  '';

  meta = with lib; {
    description = "Graph execution engine for AMD Vitis AI";
    homepage = "https://github.com/amd/graph_engine";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
