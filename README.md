# nix-amd-npu

Nix flake for AMD Ryzen AI NPU support on NixOS.

## Status

🚧 **Work in Progress** - This flake is under active development.

### Build Status

- XRT builds 596/894 targets successfully
- Blocked on: aiebu tools require static linking which conflicts with Nix's dynamic linking model
- Need to either:
  1. Patch aiebu CMake to disable static builds entirely, or
  2. Disable building the aiebu tools (they may not be needed for NPU runtime)

## What's Included

- **xrt** - Xilinx Runtime (XRT) base library
- **xrt-plugin-amdxdna** - AMD XDNA plugin for NPU access

## Requirements

- NixOS with kernel 6.14+ (has `amdxdna` driver built-in)
- AMD Ryzen AI processor (Strix Point, Krackan, etc.)
- `/dev/accel0` device present

## Usage

### Quick Test

```bash
# Try the development shell
nix develop github:robcohen/nix-amd-npu

# Check NPU detection
xrt-smi examine
```

### Add to Your Flake

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    amd-npu.url = "github:robcohen/nix-amd-npu";
  };

  outputs = { self, nixpkgs, amd-npu, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        amd-npu.nixosModules.default
        {
          hardware.amd-npu.enable = true;
        }
      ];
    };
  };
}
```

### Manual Package Installation

```nix
environment.systemPackages = [
  amd-npu.packages.x86_64-linux.xrt
  amd-npu.packages.x86_64-linux.xrt-plugin-amdxdna
];
```

## Verified Hardware

- [ ] ThinkPad P16s Gen 4 AMD (Ryzen AI 7 PRO 350, Krackan NPU)
- [ ] Framework 16 (Ryzen AI 9 HX 370)
- [ ] Other Strix Point systems

## Building from Source

```bash
git clone https://github.com/robcohen/nix-amd-npu
cd nix-amd-npu
nix build .#xrt
nix build .#xrt-plugin-amdxdna
```

## References

- [AMD xdna-driver](https://github.com/amd/xdna-driver)
- [Xilinx XRT](https://github.com/Xilinx/XRT)
- [Ryzen AI Software Docs](https://ryzenai.docs.amd.com/en/latest/linux.html)
- [Arch Linux xrt-plugin-amdxdna](https://gitlab.archlinux.org/archlinux/packaging/packages/xrt-plugin-amdxdna)

## License

Apache 2.0 (same as upstream XRT and xdna-driver)
