{ inputs, self, ... }:
{
  flake.nixosModules = {
    default = self.nixosModules.amd-npu;

    amd-npu = { config, lib, pkgs, ... }:
      with lib;
      let
        cfg = config.hardware.amd-npu;
      in
      {
        options.hardware.amd-npu = {
          enable = mkEnableOption "AMD Ryzen AI NPU support";

          package = mkOption {
            type = types.package;
            default = self.packages.${pkgs.system}.xrt-amdxdna;
            defaultText = literalExpression "inputs.nix-amd-npu.packages.\${pkgs.system}.xrt-amdxdna";
            description = "The XRT package with AMDXDNA plugin to use.";
          };
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
          environment.systemPackages = [ cfg.package ];

          # Set up environment for XRT
          environment.variables = {
            XILINX_XRT = "${cfg.package}/opt/xilinx/xrt";
          };
        };
      };
  };
}
