# Shared meta attribute helpers for nix-amd-npu
{ lib }:

{
  # Create a standard meta attribute set for AMD NPU packages
  # Usage: meta = mkMeta { description = "..."; license = lib.licenses.asl20; };
  mkMeta = {
    description,
    homepage ? null,
    license ? lib.licenses.asl20,
    platforms ? [ "x86_64-linux" ],
    # For future nixpkgs upstreaming: maintainers = with lib.maintainers; [ ... ];
    maintainers ? [ ],
    broken ? false,
    ...
  }@args:
    {
      inherit description platforms maintainers broken;
      inherit license;
    }
    // lib.optionalAttrs (homepage != null) { inherit homepage; }
    // (removeAttrs args [ "description" "homepage" "license" "platforms" "maintainers" "broken" ]);

  # Common meta for Vitis AI components
  vitisAiMeta = description: {
    inherit description;
    homepage = "https://github.com/amd";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };

  # Common meta for XRT components
  xrtMeta = description: {
    inherit description;
    homepage = "https://github.com/Xilinx/XRT";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
