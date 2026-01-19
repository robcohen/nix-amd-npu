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
, unilog
}:

stdenv.mkDerivation rec {
  pname = "xir";
  version = "3.5.0";

  src = fetchFromGitHub {
    owner = "amd";
    repo = "xir";
    rev = "402341dd389d2f3ecd128e0a414d3fae3ca42db0";
    hash = "sha256-r6ZfIsUnUwVy31tb0JYWa24rxfnVl8W4AZbxqqVfc5s=";
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
    unilog
  ];

  cmakeFlags = [
    "-DBUILD_TEST=OFF"
    "-DBUILD_PYTHON=OFF"
    "-DBUILD_DOC=OFF"
    "-DBUILD_CONTRIB=OFF"
    "-DBUILD_SHARED_LIBS=ON"
  ];

  # Fix -Werror, git version detection, and GCC 15 compatibility
  postPatch = ''
    substituteInPlace cmake/VitisCommon.cmake \
      --replace-fail "-Werror" ""

    # Create a fake git repo so the version detection works
    git init
    git config user.email "nix@build"
    git config user.name "Nix Build"
    git add -A
    git commit -m "Nix build" --allow-empty

    # Fix GCC 15 compatibility - add missing cstdint include
    for f in $(find . -name "*.cpp" -o -name "*.hpp"); do
      if grep -q 'uint64_t\|int64_t\|uint32_t\|int32_t' "$f" 2>/dev/null; then
        if ! grep -q '#include <cstdint>' "$f" 2>/dev/null; then
          sed -i '1i #include <cstdint>' "$f"
        fi
      fi
    done

    # Fix protobuf/abseil linking issue - add abseil at top-level CMakeLists
    # Newer protobuf requires explicit abseil linking
    sed -i '/find_package(Protobuf REQUIRED)/a find_package(absl REQUIRED)' CMakeLists.txt

    # Add abseil to xir library link (after protobuf)
    sed -i 's/PRIVATE protobuf::libprotobuf/PRIVATE protobuf::libprotobuf absl::hash absl::raw_hash_set absl::strings/' src/xir/CMakeLists.txt

    # Also fix tools linking
    sed -i 's/target_link_libraries(xir_util xir protobuf::libprotobuf unilog::unilog)/target_link_libraries(xir_util xir protobuf::libprotobuf unilog::unilog absl::hash absl::raw_hash_set absl::strings)/' tools/CMakeLists.txt
  '';

  meta = with lib; {
    description = "Xilinx Intermediate Representation for deep learning algorithms";
    homepage = "https://github.com/amd/xir";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
