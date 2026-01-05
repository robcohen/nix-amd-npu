{
  description = "AMD Ryzen AI NPU support for NixOS (XRT + XDNA driver)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, nixpkgs, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      imports = [
        ./parts/packages.nix
        ./parts/devshell.nix
        ./parts/nixos-module.nix
      ];

      # Overlay for nixpkgs compatibility - allows `pkgs.xrt` when applied
      flake.overlays.default = final: prev: {
        xrt = final.callPackage ./pkgs/xrt {
          pybind11 = final.python3Packages.pybind11;
        };
        xrt-plugin-amdxdna = final.callPackage ./pkgs/xrt-plugin-amdxdna {
          inherit (final) xrt;
        };
        xrt-amdxdna = final.symlinkJoin {
          name = "xrt-amdxdna-${final.xrt.version}";
          paths = [ final.xrt final.xrt-plugin-amdxdna ];
          postBuild = ''
            cd $out/opt/xilinx/xrt/lib
            pluginLib="${final.xrt-plugin-amdxdna}/opt/xilinx/xrt/lib"
            ln -sf "$pluginLib/libxrt_driver_xdna.so.2" .
            ln -sf "$pluginLib/libxrt_driver_xdna.so.${final.xrt-plugin-amdxdna.pluginVersion}" .
          '';
        };
      };

      perSystem = { system, ... }: {
        _module.args.pkgs = import nixpkgs {
          inherit system;
          # Note: XRT is Apache-2.0 licensed, no unfree components required
        };
      };
    };
}
