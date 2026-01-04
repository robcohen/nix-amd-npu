{
  description = "AMD Ryzen AI NPU support for NixOS (XRT + XDNA driver)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # XRT version from Arch packaging
        xrtVersion = "202610.2.21.21";
        xrtPluginVersion = "2.21.0";

        xrt = pkgs.callPackage ./pkgs/xrt {
          pybind11 = pkgs.python3Packages.pybind11;
        };
        xrt-plugin-amdxdna = pkgs.callPackage ./pkgs/xrt-plugin-amdxdna {
          inherit xrt;
        };

      in {
        packages = {
          inherit xrt xrt-plugin-amdxdna;
          default = xrt-plugin-amdxdna;
        };

        # Development shell with both packages
        devShells.default = pkgs.mkShell {
          packages = [
            xrt
            xrt-plugin-amdxdna
          ];

          shellHook = ''
            echo "AMD Ryzen AI NPU Development Environment"
            echo "XRT version: ${xrtVersion}"
            echo ""
            echo "To verify NPU detection:"
            echo "  xrt-smi examine"
            echo ""
            export LD_LIBRARY_PATH="${xrt}/lib:${xrt-plugin-amdxdna}/lib:$LD_LIBRARY_PATH"
            source ${xrt}/setup.sh 2>/dev/null || true
          '';
        };
      }
    ) // {
      # NixOS module for system-wide configuration
      nixosModules.default = { config, lib, pkgs, ... }:
        with lib;
        let
          cfg = config.hardware.amd-npu;
        in {
          options.hardware.amd-npu = {
            enable = mkEnableOption "AMD Ryzen AI NPU support";
          };

          config = mkIf cfg.enable {
            # Ensure amdxdna kernel module is loaded
            boot.kernelModules = [ "amdxdna" ];

            # Add udev rules for NPU device access
            services.udev.extraRules = ''
              # AMD NPU (amdxdna) - allow users in video group
              SUBSYSTEM=="accel", KERNEL=="accel[0-9]*", GROUP="video", MODE="0660"
            '';

            # Add XRT and plugin to system packages
            environment.systemPackages = [
              self.packages.${pkgs.system}.xrt
              self.packages.${pkgs.system}.xrt-plugin-amdxdna
            ];

            # Set up library paths
            environment.variables = {
              XILINX_XRT = "${self.packages.${pkgs.system}.xrt}";
            };
          };
        };
    };
}
