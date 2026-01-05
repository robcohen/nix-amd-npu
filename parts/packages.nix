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

          pluginLib="${xrt-plugin-amdxdna}/opt/xilinx/xrt/lib"
          if [ ! -f "$pluginLib/libxrt_driver_xdna.so.2" ]; then
            echo "ERROR: Plugin library not found at $pluginLib/libxrt_driver_xdna.so.2"
            exit 1
          fi

          ln -sf "$pluginLib/libxrt_driver_xdna.so.2" .
          ln -sf "$pluginLib/libxrt_driver_xdna.so.${xrt-plugin-amdxdna.pluginVersion}" .
        '';
      };
    in
    {
      packages = {
        inherit xrt xrt-plugin-amdxdna xrt-amdxdna;
        default = xrt-amdxdna;
      };

      # Integration tests - run with `nix flake check`
      checks = {
        # Verify XRT builds and has expected binaries
        xrt-binaries = pkgs.runCommand "check-xrt-binaries" {} ''
          echo "Checking XRT binaries..."
          test -x ${xrt}/bin/xrt-smi || (echo "FAIL: xrt-smi not found" && exit 1)
          test -x ${xrt}/bin/xbutil || (echo "FAIL: xbutil not found" && exit 1)
          echo "PASS: XRT binaries present"
          touch $out
        '';

        # Verify plugin library exists and has correct soname
        plugin-library = pkgs.runCommand "check-plugin-library" {} ''
          echo "Checking plugin library..."
          pluginLib="${xrt-plugin-amdxdna}/opt/xilinx/xrt/lib"
          test -f "$pluginLib/libxrt_driver_xdna.so.2" || (echo "FAIL: plugin .so.2 not found" && exit 1)
          test -f "$pluginLib/libxrt_driver_xdna.so.${xrt-plugin-amdxdna.pluginVersion}" || (echo "FAIL: plugin .so.${xrt-plugin-amdxdna.pluginVersion} not found" && exit 1)
          echo "PASS: Plugin library present with correct versions"
          touch $out
        '';

        # Verify combined package has plugin discoverable by XRT
        plugin-discovery = pkgs.runCommand "check-plugin-discovery" {} ''
          echo "Checking plugin discovery in combined package..."
          xrtLib="${xrt-amdxdna}/opt/xilinx/xrt/lib"
          test -L "$xrtLib/libxrt_driver_xdna.so.2" || (echo "FAIL: plugin symlink not in combined package" && exit 1)
          # Verify symlink resolves correctly
          readlink -f "$xrtLib/libxrt_driver_xdna.so.2" > /dev/null || (echo "FAIL: plugin symlink broken" && exit 1)
          echo "PASS: Plugin discoverable in combined package"
          touch $out
        '';

        # Verify pkg-config files are generated
        pkg-config-files = pkgs.runCommand "check-pkg-config" {} ''
          echo "Checking pkg-config files..."
          test -f ${xrt}/lib/pkgconfig/xrt.pc || (echo "FAIL: xrt.pc not found" && exit 1)
          test -f ${xrt-plugin-amdxdna}/lib/pkgconfig/xrt-amdxdna.pc || (echo "FAIL: xrt-amdxdna.pc not found" && exit 1)
          echo "PASS: pkg-config files present"
          touch $out
        '';

        # Verify environment setup works
        environment-setup = pkgs.runCommand "check-environment" {} ''
          echo "Checking environment setup..."
          export XILINX_XRT="${xrt-amdxdna}/opt/xilinx/xrt"
          test -d "$XILINX_XRT" || (echo "FAIL: XILINX_XRT directory doesn't exist" && exit 1)
          test -d "$XILINX_XRT/lib" || (echo "FAIL: XRT lib directory doesn't exist" && exit 1)
          test -d "$XILINX_XRT/bin" || (echo "FAIL: XRT bin directory doesn't exist" && exit 1)
          echo "PASS: Environment directories valid"
          touch $out
        '';
      };
    };
}
