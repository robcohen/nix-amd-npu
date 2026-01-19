{ inputs, self, ... }:
{
  flake.nixosModules = {
    default = self.nixosModules.amd-npu;

    # NixOS module for AMD NPU support
    # Works both as a flake module and when copied to nixpkgs
    amd-npu = { config, lib, pkgs, ... }:
      let
        cfg = config.hardware.amd-npu;
      in
      {
        options.hardware.amd-npu = {
          enable = lib.mkEnableOption "AMD Ryzen AI NPU support";

          package = lib.mkOption {
            type = lib.types.package;
            default = pkgs.xrt-amdxdna or self.packages.${pkgs.system}.xrt-amdxdna;
            defaultText = lib.literalExpression "pkgs.xrt-amdxdna";
            description = "The XRT package with AMDXDNA plugin to use.";
          };

          group = lib.mkOption {
            type = lib.types.str;
            default = "video";
            description = "Group granted access to NPU device and elevated memlock limits.";
          };

          enableDevTools = lib.mkOption {
            type = lib.types.bool;
            default = true;
            description = ''
              Whether to include XRT development tools (xrt-smi, xclbinutil) in the system path.
              Disable this for minimal production installations.
            '';
          };

          memlockLimit = lib.mkOption {
            type = lib.types.either lib.types.ints.positive (lib.types.enum [ "unlimited" ]);
            default = "unlimited";
            description = ''
              Memory lock limit for NPU buffer allocation.
              The NPU driver needs to mmap large buffers (64MB+).
              Set to "unlimited" or a specific number of bytes.
            '';
          };

          kernelModule = lib.mkOption {
            type = lib.types.str;
            default = "amdxdna";
            description = "Kernel module name for the AMD NPU driver.";
          };

          extraUdevRules = lib.mkOption {
            type = lib.types.lines;
            default = "";
            description = "Additional udev rules for NPU device configuration.";
          };
        };

        config = lib.mkIf cfg.enable {
          # NOTE: Consumers must apply self.overlays.default to their pkgs
          # for pkgs.xrt-amdxdna to be available. Setting nixpkgs.overlays
          # here causes infinite recursion because evaluating `pkgs` requires
          # nixpkgs.overlays, which requires this module, which needs pkgs.

          # Ensure amdxdna kernel module is loaded
          boot.kernelModules = [ cfg.kernelModule ];

          # SVA (Shared Virtual Addressing) requires IOMMU translated mode, not passthrough
          boot.kernelParams = [ "iommu.passthrough=0" ];

          # Add udev rules for NPU device access
          services.udev.extraRules = ''
            # AMD NPU (amdxdna) - allow users in ${cfg.group} group
            SUBSYSTEM=="accel", KERNEL=="accel[0-9]*", GROUP="${cfg.group}", MODE="0660"
            ${cfg.extraUdevRules}
          '';

          # Increase locked memory limit for NPU buffer allocation
          # The NPU driver needs to mmap large buffers (64MB+)
          # Only grant to members of the configured group (default: video)
          security.pam.loginLimits = [
            { domain = "@${cfg.group}"; type = "soft"; item = "memlock"; value = toString cfg.memlockLimit; }
            { domain = "@${cfg.group}"; type = "hard"; item = "memlock"; value = toString cfg.memlockLimit; }
          ];

          # Add combined XRT+plugin to system packages
          environment.systemPackages = lib.mkIf cfg.enableDevTools [ cfg.package ];

          # Set up environment for XRT
          environment.variables = {
            XILINX_XRT = "${cfg.package}/opt/xilinx/xrt";
          };

          # Assertions to help users diagnose issues
          assertions = [
            {
              assertion = config.boot.kernelPackages.kernelAtLeast "6.10";
              message = ''
                AMD NPU support requires kernel 6.10 or newer.
                The amdxdna driver is in mainline kernel starting from 6.14.
                For kernels 6.10-6.13, you may need to build the driver separately.
              '';
            }
          ];

          # Warnings for common configuration issues
          warnings = lib.optional
            (cfg.group != "video" && !(builtins.elem cfg.group config.users.groups))
            "hardware.amd-npu.group is set to '${cfg.group}' but this group doesn't exist. NPU access may not work.";
        };
      };
  };
}
