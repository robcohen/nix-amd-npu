{ lib
, stdenv
, fetchFromGitHub
, cmake
, ninja
, pkg-config
, git
, protobuf
, abseil-cpp
, unilog
, xir
}:

stdenv.mkDerivation rec {
  pname = "target-factory";
  version = "3.5.0";

  src = fetchFromGitHub {
    owner = "amd";
    repo = "target_factory";
    rev = "7270acd2eb90e8f5b4fc7354c66c2a3fd960111f";
    hash = "sha256-GF8x4a0+5zGRlu6wrMN8YYSvl1iybNvCFjLYmb1jm3M=";
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
    unilog
    xir
  ];

  cmakeFlags = [
    "-DBUILD_TEST=OFF"
    "-DBUILD_PYTHON=OFF"
    "-DBUILD_SHARED_LIBS=ON"
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

    # Fix GCC 15 compatibility - add missing cstdint include
    for f in $(find . -name "*.cpp" -o -name "*.hpp"); do
      if grep -q 'uint64_t\|int64_t\|uint32_t\|int32_t' "$f" 2>/dev/null; then
        if ! grep -q '#include <cstdint>' "$f" 2>/dev/null; then
          sed -i '1i #include <cstdint>' "$f"
        fi
      fi
    done

    # Fix protobuf/abseil linking
    sed -i '/find_package(Protobuf REQUIRED)/a find_package(absl REQUIRED)' CMakeLists.txt
  '';

  meta = with lib; {
    description = "Factory to generate DPU target description file for AMD Vitis AI";
    homepage = "https://github.com/amd/target_factory";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
