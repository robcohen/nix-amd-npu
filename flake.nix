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
        # XRT and XDNA driver
        xrt = final.callPackage ./pkgs/xrt { };
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

        # Vitis AI libraries
        unilog = final.callPackage ./pkgs/vitis-ai/unilog { };
        xir = final.callPackage ./pkgs/vitis-ai/xir {
          inherit (final) unilog;
        };
        target-factory = final.callPackage ./pkgs/vitis-ai/target-factory {
          inherit (final) unilog xir;
        };
        vart = final.callPackage ./pkgs/vitis-ai/vart {
          inherit (final) unilog xir target-factory;
          xrt = null;
        };
        trace-logging = final.callPackage ./pkgs/vitis-ai/trace-logging { };
        graph-engine = final.callPackage ./pkgs/vitis-ai/graph-engine {
          inherit (final) unilog xir vart xrt;
        };
        xaiengine = final.callPackage ./pkgs/vitis-ai/xaiengine { };
        dynamic-dispatch = final.callPackage ./pkgs/vitis-ai/dynamic-dispatch {
          inherit (final) xaiengine xrt;
        };

        # ONNX Runtime with VitisAI EP
        onnxruntime-vitisai = final.callPackage ./pkgs/onnxruntime-vitisai {
          inherit (final) xrt;
          inherit (prev) onnxruntime;
        };

        # MLIR-AIE for NPU kernel development
        mlir-aie = final.callPackage ./pkgs/mlir-aie { };

        # Whisper-IRON speech recognition
        whisper-iron = final.callPackage ./pkgs/whisper-iron {
          inherit (final) mlir-aie xrt-amdxdna;
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
