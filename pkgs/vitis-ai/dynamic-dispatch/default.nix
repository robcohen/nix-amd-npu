{ lib
, stdenv
, fetchFromGitHub
, cmake
, ninja
, pkg-config
, git
, python3
, protobuf
, abseil-cpp
, nlohmann_json
, spdlog
, zlib
, libuuid
, xaiengine
, xrt
}:

stdenv.mkDerivation rec {
  pname = "dynamic-dispatch";
  version = "1.2.0";

  src = fetchFromGitHub {
    owner = "amd";
    repo = "DynamicDispatch";
    rev = "b3051f03e20aab237cda3bbe4cd2081f76b72b06";
    hash = "sha256-kSRBn5ozQ6Ob/4qsRFvCQUHgnkqFASw/RUXQEJlGUdU=";
  };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    git
    python3
    protobuf
  ];

  buildInputs = [
    protobuf
    abseil-cpp
    nlohmann_json
    spdlog
    zlib
    libuuid
    xaiengine
    xrt
  ];

  cmakeFlags = [
    "-DBUILD_TEST=OFF"
    "-DENABLE_DD_TESTS=OFF"
    "-DENABLE_DD_PYTHON=OFF"
    "-DBUILD_SHARED_LIBS=ON"
    "-DDD_DISABLE_AIEBU=ON"
    "-DDISABLE_LARGE_TXN_OPS=ON"
    # Point to XRT
    "-DXRT_DIR=${xrt}/opt/xilinx/xrt/share/cmake/XRT"
    "-DCMAKE_PREFIX_PATH=${xrt}/opt/xilinx/xrt"
    # Help CMake find zlib
    "-DZLIB_ROOT=${zlib}"
  ];

  postPatch = ''
    # Create a fake git repo for version detection
    git init
    git config user.email "nix@build"
    git config user.name "Nix Build"
    git add -A
    git commit -m "Nix build" --allow-empty

    # Disable static zlib requirement - use system shared lib
    substituteInPlace cmake/zlib_dep.cmake \
      --replace-fail 'set(ZLIB_USE_STATIC_LIBS ON)' 'set(ZLIB_USE_STATIC_LIBS OFF)'

    # Remove pm_load op - requires internal AMD xaiengine APIs not in public repo
    sed -i '/ops\/pm_load\/pm_load.cpp/d' src/CMakeLists.txt
    rm -rf src/ops/pm_load

    # Fix protobuf 32+ API change (field renamed)
    substituteInPlace src/fusion_rt/metastate_api.cpp \
      --replace-fail 'always_print_primitive_fields' 'always_print_fields_with_no_presence'

    # Remove PM_LOAD case from txn_utils - uses internal AMD APIs
    sed -i '/case (XAIE_IO_LOAD_PM_START): {/,/^  }$/d' src/txn/txn_utils.cpp

    # Fix GCC 15 compatibility - add missing includes
    for f in $(find . -name "*.cpp" -o -name "*.hpp" -o -name "*.h"); do
      if grep -q 'uint64_t\|int64_t\|uint32_t\|int32_t' "$f" 2>/dev/null; then
        if ! grep -q '#include <cstdint>' "$f" 2>/dev/null; then
          sed -i '1i #include <cstdint>' "$f"
        fi
      fi
    done
  '';

  meta = with lib; {
    description = "Dynamic operator dispatch for AMD Ryzen AI";
    homepage = "https://github.com/amd/DynamicDispatch";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
