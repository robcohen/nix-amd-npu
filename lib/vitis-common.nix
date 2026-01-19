# Shared build helpers for Vitis AI and XRT packages
{ lib }:

rec {
  # Fix GCC 15 compatibility by adding missing cstdint includes
  # Usage: postPatch = addGcc15Compat + ''...other patches...'';
  addGcc15Compat = ''
    # Fix GCC 15 compatibility - add missing cstdint include
    for f in $(find . -name "*.cpp" -o -name "*.hpp" -o -name "*.h" 2>/dev/null); do
      if grep -q 'uint64_t\|int64_t\|uint32_t\|int32_t\|uint8_t\|int8_t\|uint16_t\|int16_t' "$f" 2>/dev/null; then
        if ! grep -q '#include <cstdint>' "$f" 2>/dev/null; then
          sed -i '1i #include <cstdint>' "$f"
        fi
      fi
    done
  '';

  # Create a fake git repo for version detection in CMake builds
  # Many Vitis AI packages use git describe for versioning
  fakeGitRepo = ''
    # Create a fake git repo so version detection works
    git init
    git config user.email "nix@build"
    git config user.name "Nix Build"
    git add -A
    git commit -m "Nix build" --allow-empty
  '';

  # Remove -Werror from VitisCommon.cmake
  # Usage: postPatch = removeWerror + ''...other patches...'';
  removeWerror = ''
    # Remove -Werror to allow builds with newer GCC
    if [ -f cmake/VitisCommon.cmake ]; then
      substituteInPlace cmake/VitisCommon.cmake \
        --replace-fail "-Werror" "" || true
    fi
    # Also remove from any CMakeLists.txt
    find . -name "CMakeLists.txt" -exec sed -i 's/-Werror//g' {} \; 2>/dev/null || true
  '';

  # Fix protobuf/abseil linking for newer protobuf versions
  # Usage: postPatch = fixProtobufAbseil + ''...other patches...'';
  fixProtobufAbseil = ''
    # Fix protobuf/abseil linking issue - newer protobuf requires explicit abseil
    if [ -f CMakeLists.txt ]; then
      if grep -q "find_package(Protobuf" CMakeLists.txt; then
        sed -i '/find_package(Protobuf REQUIRED)/a find_package(absl REQUIRED)' CMakeLists.txt || true
      fi
    fi
  '';

  # Common CMake flags for Vitis AI packages
  commonVitisFlags = [
    "-DBUILD_TEST=OFF"
    "-DBUILD_PYTHON=OFF"
    "-DBUILD_DOC=OFF"
    "-DBUILD_SHARED_LIBS=ON"
  ];

  # Create a standard Vitis AI derivation with common settings
  # This is a helper function that can be used in callPackage
  mkVitisDerivation = {
    stdenv,
    fetchFromGitHub,
    cmake,
    ninja,
    pkg-config,
    git ? null,
    ...
  }@pkgArgs: {
    pname,
    version,
    src,
    nativeBuildInputs ? [ ],
    buildInputs ? [ ],
    cmakeFlags ? [ ],
    postPatch ? "",
    meta,
    ...
  }@args:
    stdenv.mkDerivation (
      {
        inherit pname version src meta;

        nativeBuildInputs = [
          cmake
          ninja
          pkg-config
        ] ++ lib.optional (git != null) git ++ nativeBuildInputs;

        inherit buildInputs;

        cmakeFlags = commonVitisFlags ++ cmakeFlags;

        postPatch = removeWerror + addGcc15Compat + lib.optionalString (git != null) fakeGitRepo + postPatch;
      }
      // (removeAttrs args [
        "pname"
        "version"
        "src"
        "nativeBuildInputs"
        "buildInputs"
        "cmakeFlags"
        "postPatch"
        "meta"
      ])
    );
}
