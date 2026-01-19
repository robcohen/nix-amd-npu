{ lib
, stdenv
, fetchFromGitHub
, cmake
, ninja
}:

stdenv.mkDerivation rec {
  pname = "trace-logging";
  version = "1.0.0";

  src = fetchFromGitHub {
    owner = "amd";
    repo = "trace_logging";
    rev = "8f3785a9aa0241d7dc5e8300bae5ce5cbc171dbe";
    hash = "sha256-KgD3o0dEwKkZG/EsKjO9imY8fDu9eX383DTqGtW+cN0=";
  };

  nativeBuildInputs = [
    cmake
    ninja
  ];

  cmakeFlags = [
    "-DBUILD_TEST=OFF"
    "-DBUILD_SHARED_LIBS=ON"
  ];

  meta = with lib; {
    description = "Trace logging library for AMD Vitis AI";
    homepage = "https://github.com/amd/trace_logging";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
