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
        };

        config = lib.mkIf cfg.enable {
          # Apply overlay if xrt-amdxdna not already in pkgs (flake usage)
          nixpkgs.overlays = lib.mkIf (!(pkgs ? xrt-amdxdna)) [
            self.overlays.default
          ];

          # Ensure amdxdna kernel module is loaded
          boot.kernelModules = [ "amdxdna" ];

          # Add udev rules for NPU device access
          services.udev.extraRules = ''
            # AMD NPU (amdxdna) - allow users in ${cfg.group} group
            SUBSYSTEM=="accel", KERNEL=="accel[0-9]*", GROUP="${cfg.group}", MODE="0660"
          '';

          # Increase locked memory limit for NPU buffer allocation
          # The NPU driver needs to mmap large buffers (64MB+)
          # Only grant to members of the configured group (default: video)
          security.pam.loginLimits = [
            { domain = "@${cfg.group}"; type = "soft"; item = "memlock"; value = "unlimited"; }
            { domain = "@${cfg.group}"; type = "hard"; item = "memlock"; value = "unlimited"; }
          ];

          # Add combined XRT+plugin to system packages
          environment.systemPackages = [ cfg.package ];

          # Set up environment for XRT
          environment.variables = {
            XILINX_XRT = "${cfg.package}/opt/xilinx/xrt";
          };
        };
      };
  };
}
