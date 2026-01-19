# Shared library functions for nix-amd-npu
{ lib }:

rec {
  # Import sub-modules
  versions = import ./versions.nix;
  meta = import ./meta.nix { inherit lib; };
  vitisCommon = import ./vitis-common.nix { inherit lib; };

  # Re-export commonly used functions
  inherit (meta) mkMeta;
  inherit (vitisCommon) mkVitisDerivation addGcc15Compat fakeGitRepo removeWerror;
}
