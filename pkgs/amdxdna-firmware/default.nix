{ lib
, fetchurl
, stdenvNoCC
}:

let
  # Firmware metadata from xdna-driver tools/info.json (v2.21.75)
  #
  # Upstream firmware source: https://gitlab.com/kernel-firmware/drm-firmware/-/tree/amd-ipu-staging/amdnpu
  #
  # Hardware device IDs (PCI vendor 0x1002):
  #   1502_00  - Hawk Point (Ryzen 8040 series)
  #   17f0_10  - Strix Point (Ryzen AI 300 series, sub-rev 10)
  #   17f0_11  - Strix Point (Ryzen AI 300 series, sub-rev 11)
  #   17f1_10  - Krackan Point / Ryzen AI Max (sub-rev 10)
  #   17f1_13  - Krackan Point / Ryzen AI Max (sub-rev 13)
  #   17f1_14  - Krackan Point / Ryzen AI Max (sub-rev 14)
  #
  # NOTE: The old 17f0_00 directory was removed upstream in July 2026.
  # Firmware was reorganized into sub-revision-specific directories with a
  # new naming scheme. The file format shifted from npu.sbin.<version> to
  # either <rai>_npu.sbin.<version> (17f0_*) or npu.sbin.<version> (17f1_*).
  firmwares = [
    {
      name = "npu1";
      url = "https://gitlab.com/kernel-firmware/drm-firmware/-/raw/amd-ipu-staging/amdnpu/1502_00/1.5_npu.sbin.1.5.5.391";
      hash = "sha256-0T/5+5XGzqQCE/pp5aNGVSnwC7Z8CYTWI0PG4xgI+54=";
      installDir = "amdnpu/1502_00";
    }
    {
      name = "npu2";
      url = "https://gitlab.com/kernel-firmware/drm-firmware/-/raw/amd-ipu-staging/amdnpu/17f0_10/1.7_npu.sbin.1.1.2.64";
      hash = "sha256-ftDyQvJTtYHrdcjIs0Y42S+6WYBjtjWlQb65SGkkVGk=";
      installDir = "amdnpu/17f0_10";
    }
    {
      name = "npu3";
      url = "https://gitlab.com/kernel-firmware/drm-firmware/-/raw/amd-ipu-staging/amdnpu/17f0_11/1.7_npu.sbin.1.1.2.65";
      hash = "sha256-PjyZbvHlYulu5MTZD6qfrxEyxy2jrxvPNdWSzDSQP+0=";
      installDir = "amdnpu/17f0_11";
    }
    {
      name = "npu4";
      url = "https://gitlab.com/kernel-firmware/drm-firmware/-/raw/amd-ipu-staging/amdnpu/17f1_10/npu.sbin.2.11.0.51";
      hash = "sha256-IkF3MKY4yKmIq03SzKrLWrYBjOmwaCFrEpZePKGRqCU=";
      installDir = "amdnpu/17f1_10";
    }
    {
      name = "npu5";
      url = "https://gitlab.com/kernel-firmware/drm-firmware/-/raw/amd-ipu-staging/amdnpu/17f1_13/npu.sbin.2.11.0.51";
      hash = "sha256-kE2tzb9iE6veXPM1qTiHNAL/1tOpsSgGF1CpbgGCsG0=";
      installDir = "amdnpu/17f1_13";
    }
    {
      name = "npu6";
      url = "https://gitlab.com/kernel-firmware/drm-firmware/-/raw/amd-ipu-staging/amdnpu/17f1_14/npu.sbin.2.11.0.51";
      hash = "sha256-3N9LFszF9i7dt76wBJE+9fZ2mg5r9FBflR3FYrI2UI4=";
      installDir = "amdnpu/17f1_14";
    }
  ];

  sources = map (fw: {
    inherit (fw) name installDir;
    src = fetchurl {
      inherit (fw) url hash;
      name = "${fw.name}-npu.dev.sbin";
    };
  }) firmwares;

in stdenvNoCC.mkDerivation {
  pname = "amdxdna-firmware";
  version = "2.21.75";

  dontUnpack = true;

  installPhase = ''
    runHook preInstall
  '' + lib.concatMapStringsSep "\n" (fw: ''
    install -Dm444 ${fw.src} $out/lib/firmware/${fw.installDir}/npu.dev.sbin
  '') sources + ''

    runHook postInstall
  '';

  meta = with lib; {
    description = "Development firmware for AMD XDNA NPU (required by out-of-tree driver)";
    homepage = "https://gitlab.com/kernel-firmware/drm-firmware/-/tree/amd-ipu-staging/amdnpu";
    license = licenses.unfreeRedistributableFirmware;
    platforms = [ "x86_64-linux" ];
  };
}
