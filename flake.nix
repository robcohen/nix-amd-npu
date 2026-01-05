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

        # Combined XRT with XDNA plugin - plugins must be in same lib dir for discovery
        xrt-amdxdna = pkgs.symlinkJoin {
          name = "xrt-amdxdna-${xrt.version}";
          paths = [ xrt xrt-plugin-amdxdna ];
          postBuild = ''
            # Ensure the plugin is discoverable by XRT
            # XRT looks for libxrt_driver_*.so.MAJOR_VERSION in its lib dir
            cd $out/opt/xilinx/xrt/lib
            ln -sf ${xrt-plugin-amdxdna}/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2 . 2>/dev/null || true
            ln -sf ${xrt-plugin-amdxdna}/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2.21.0 . 2>/dev/null || true
          '';
        };

      in {
        packages = {
          inherit xrt xrt-plugin-amdxdna xrt-amdxdna;
          default = xrt-amdxdna;
        };

        # Development shell with combined package
        devShells.default = pkgs.mkShell {
          packages = [
            xrt-amdxdna
          ];

          shellHook = ''
            echo "AMD Ryzen AI NPU Development Environment"
            echo "XRT version: ${xrtVersion}"
            echo ""
            echo "To verify NPU detection:"
            echo "  xrt-smi examine"
            echo ""
            export XILINX_XRT="${xrt-amdxdna}/opt/xilinx/xrt"
            export LD_LIBRARY_PATH="${xrt-amdxdna}/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH"
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

            # Increase locked memory limit for NPU buffer allocation
            # The NPU driver needs to mmap large buffers (64MB+)
            security.pam.loginLimits = [
              { domain = "*"; type = "soft"; item = "memlock"; value = "unlimited"; }
              { domain = "*"; type = "hard"; item = "memlock"; value = "unlimited"; }
            ];

            # Add combined XRT+plugin to system packages
            environment.systemPackages = [
              self.packages.${pkgs.system}.xrt-amdxdna
            ];

            # Set up environment for XRT
            environment.variables = {
              XILINX_XRT = "${self.packages.${pkgs.system}.xrt-amdxdna}/opt/xilinx/xrt";
            };
          };
        };
    };
}
