{ lib
, stdenv
, fetchFromGitHub
, cmake
, ninja
, pkg-config
, git
, protobuf
, abseil-cpp
, boost
, glog
, unilog
, xir
, target-factory
, xrt ? null
}:

stdenv.mkDerivation rec {
  pname = "vart";
  version = "3.5.0";

  src = fetchFromGitHub {
    owner = "amd";
    repo = "vart";
    rev = "dafd687831e817720f430e6f6033f5f2cd78fe5b";
    hash = "sha256-nzt9D9jC7V2DSWHYZbmxesCmZN2Gc3CFanzpM0T6lUI=";
  };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    git
    protobuf
  ];

  buildInputs = [
    protobuf
    abseil-cpp
    boost
    glog
    unilog
    xir
    target-factory
  ] ++ lib.optionals (xrt != null) [ xrt ];

  cmakeFlags = [
    "-DBUILD_TEST=OFF"
    "-DBUILD_PYTHON=OFF"
    "-DBUILD_SHARED_LIBS=ON"
    "-DENABLE_CPU_RUNNER=OFF"
    "-DENABLE_SIM_RUNNER=OFF"
    "-DENABLE_DPU_RUNNER=OFF"
    "-DENABLE_XRNN_RUNNER=OFF"
  ];

  postPatch = ''
    substituteInPlace cmake/VitisCommon.cmake \
      --replace-fail "-Werror" ""

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
      # Add unistd.h for syscall, close, etc.
      if grep -qE 'syscall|[^a-z]close\(' "$f" 2>/dev/null; then
        if ! grep -q '#include <unistd.h>' "$f" 2>/dev/null; then
          sed -i '1i #include <unistd.h>' "$f"
        fi
      fi
      # Add sys/syscall.h for SYS_* constants
      if grep -q 'SYS_' "$f" 2>/dev/null; then
        if ! grep -q '#include <sys/syscall.h>' "$f" 2>/dev/null; then
          sed -i '1i #include <sys/syscall.h>' "$f"
        fi
      fi
    done

    # Fix protobuf/abseil linking
    sed -i '/find_package(Protobuf REQUIRED)/a find_package(absl REQUIRED)' CMakeLists.txt

    # Remove conda path that breaks builds
    sed -i '/link_directories.*CONDA_PREFIX/d' CMakeLists.txt
  '';

  meta = with lib; {
    description = "Vitis AI Runtime for AMD accelerators";
    homepage = "https://github.com/amd/vart";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
