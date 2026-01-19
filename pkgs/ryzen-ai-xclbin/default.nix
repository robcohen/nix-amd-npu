{ lib
, stdenv
, requireFile
}:

# AMD Ryzen AI xclbin files - NPU binary configurations
#
# These are pre-compiled NPU kernel binaries required for certain devices:
# - PHX (Phoenix) / HPT (Hawk Point): Requires 4x4.xclbin
# - STX (Strix) / KRK: Uses built-in binaries (no xclbin needed for INT8 models)
#
# The xclbin files are bundled with the Ryzen AI Software distribution.

stdenv.mkDerivation rec {
  pname = "ryzen-ai-xclbin";
  version = "1.6.1";

  src = requireFile {
    name = "ryzen_ai-${version}.tgz";
    url = "https://www.amd.com/en/developer/resources/ryzen-ai-software.html";
    hash = "sha256-0000000000000000000000000000000000000000000=";  # Update after download
    message = ''
      AMD Ryzen AI xclbin files require the Ryzen AI Software distribution.

      1. Visit: https://www.amd.com/en/developer/resources/ryzen-ai-software.html
      2. Register for AMD Early Access Lounge
      3. Download ryzen_ai-${version}.tgz
      4. Add to Nix store: nix-store --add-fixed sha256 ryzen_ai-${version}.tgz
    '';
  };

  dontBuild = true;

  unpackPhase = ''
    mkdir -p source
    tar -xzf $src -C source
  '';

  installPhase = ''
    mkdir -p $out/share/xclbin/{phoenix,strix}

    # Phoenix/Hawk Point xclbin
    if [ -f source/xclbins/phoenix/4x4.xclbin ]; then
      cp source/xclbins/phoenix/4x4.xclbin $out/share/xclbin/phoenix/
    fi

    # Strix xclbin (if needed)
    if [ -f source/xclbins/strix/AMD_AIE2P_4x4_Overlay.xclbin ]; then
      cp source/xclbins/strix/AMD_AIE2P_4x4_Overlay.xclbin $out/share/xclbin/strix/
    fi

    # Create symlinks for common access
    if [ -f $out/share/xclbin/phoenix/4x4.xclbin ]; then
      ln -s phoenix/4x4.xclbin $out/share/xclbin/default-phx.xclbin
    fi
    if [ -f $out/share/xclbin/strix/AMD_AIE2P_4x4_Overlay.xclbin ]; then
      ln -s strix/AMD_AIE2P_4x4_Overlay.xclbin $out/share/xclbin/default-stx.xclbin
    fi

    # Configuration info
    cat > $out/share/xclbin/README.txt << 'EOF'
    AMD Ryzen AI NPU Binary Configurations (xclbin)

    PHX (Phoenix) / HPT (Hawk Point):
      - Use: phoenix/4x4.xclbin
      - Set provider option to: $out/share/xclbin/phoenix/4x4.xclbin

    STX (Strix) / KRK:
      - For INT8 models: No xclbin needed (uses built-in)
      - For other models: strix/AMD_AIE2P_4x4_Overlay.xclbin

    Note: STX/KRK devices running INT8 models should use target=X2 (default)
          without specifying an xclbin path.
    EOF
  '';

  meta = with lib; {
    description = "AMD Ryzen AI xclbin files - NPU binary configurations";
    homepage = "https://www.amd.com/en/developer/resources/ryzen-ai-software.html";
    license = licenses.unfree;  # AMD proprietary
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
