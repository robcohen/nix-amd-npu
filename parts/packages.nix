{ inputs, ... }:
{
  perSystem = { config, pkgs, system, ... }:
    let
      xrt = pkgs.callPackage ../pkgs/xrt {
        pybind11 = pkgs.python3Packages.pybind11;
      };

      xrt-plugin-amdxdna = pkgs.callPackage ../pkgs/xrt-plugin-amdxdna {
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
    in
    {
      packages = {
        inherit xrt xrt-plugin-amdxdna xrt-amdxdna;
        default = xrt-amdxdna;
      };
    };
}
