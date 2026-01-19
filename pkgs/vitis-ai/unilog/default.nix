{ lib
, stdenv
, fetchFromGitHub
, cmake
, ninja
, pkg-config
, glog
, boost
}:

stdenv.mkDerivation rec {
  pname = "unilog";
  version = "3.5.0";

  src = fetchFromGitHub {
    owner = "amd";
    repo = "unilog";
    rev = "3abf8046d7ec8e651b8ec7ef19627a667ffaa741";
    hash = "sha256-TAsl/bCVwgVvbz3dQ9EKfBgZJArz4K2bae1hP/HuH3Q=";
  };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
  ];

  buildInputs = [
    glog
    boost
  ];

  # Propagate glog and boost so dependents can find them
  propagatedBuildInputs = [
    glog
    boost
  ];

  cmakeFlags = [
    "-DBUILD_TEST=OFF"
    "-DBUILD_PYTHON=OFF"
    "-DBUILD_SHARED_LIBS=ON"
  ];

  # Fix -Werror and cmake config issues
  postPatch = ''
    substituteInPlace cmake/VitisCommon.cmake \
      --replace-fail "-Werror" ""

    # Fix config.cmake.in to not fallback to pkg-config for glog
    # The glog from nixpkgs provides cmake config, not pkg-config
    substituteInPlace cmake/config.cmake.in \
      --replace-fail 'if(NOT glog_FOUND)
  message(STATUS "cannot find glogConfig.cmake fallback to pkg-config")
  find_package(PkgConfig)
  pkg_search_module(PKG_GLOG REQUIRED IMPORTED_TARGET GLOBAL libglog)
  add_library(glog::glog ALIAS PkgConfig::PKG_GLOG)
endif(NOT glog_FOUND)' 'if(NOT glog_FOUND)
  message(FATAL_ERROR "glog not found - ensure glog cmake config is available")
endif(NOT glog_FOUND)'

    # Fix glog API compatibility - LogMessageVoidify is not exposed in newer glog
    # Add our own void cast helper to the header
    substituteInPlace include/UniLog/UniLog.hpp \
      --replace-fail '#include <glog/logging.h>' '#include <glog/logging.h>

// Compatibility shim for newer glog versions where LogMessageVoidify is not exposed
#ifndef GOOGLE_GLOG_LOGMESSAGEVOIDIFY_DEFINED
namespace google {
class LogMessageVoidify {
 public:
  void operator&(std::ostream&) {}
};
}
#define GOOGLE_GLOG_LOGMESSAGEVOIDIFY_DEFINED
#endif'
  '';

  meta = with lib; {
    description = "Unified logging library for AMD Vitis AI";
    homepage = "https://github.com/amd/unilog";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
