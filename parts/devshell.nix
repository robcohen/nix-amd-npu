{ inputs, ... }:
{
  perSystem = { config, pkgs, ... }:
    let
      xrt = config.packages.xrt;
      xrt-amdxdna = config.packages.xrt-amdxdna;
    in
    {
      devShells.default = pkgs.mkShell {
        packages = [
          xrt-amdxdna
        ];

        shellHook = ''
          echo "AMD Ryzen AI NPU Development Environment"
          echo "XRT version: ${xrt.version}"
          echo ""
          echo "To verify NPU detection:"
          echo "  xrt-smi examine"
          echo ""
          export XILINX_XRT="${xrt-amdxdna}/opt/xilinx/xrt"
          export LD_LIBRARY_PATH="${xrt-amdxdna}/opt/xilinx/xrt/lib:''${LD_LIBRARY_PATH:-}"
        '';
      };
    };
}
