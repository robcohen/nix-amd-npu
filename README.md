# nix-amd-npu

Nix flake for AMD Ryzen AI NPU support on NixOS.

## Status

- XRT builds successfully (726 targets)
- xrt-plugin-amdxdna builds successfully (23 targets)
- NPU detection works with proper system configuration

## What's Included

| Package | Description |
|---------|-------------|
| `xrt` | Xilinx Runtime (XRT) base library |
| `xrt-plugin-amdxdna` | AMD XDNA shim plugin for NPU access |
| `xrt-amdxdna` | Combined package (default) |

## Requirements

- NixOS with kernel 6.14+ (has `amdxdna` driver built-in)
- AMD Ryzen AI processor (Strix Point, Krackan, etc.)
- `/dev/accel0` device present
- Increased memlock limit (configured automatically by the NixOS module)

## Usage

### NixOS Module (Recommended)

Add to your flake:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nix-amd-npu.url = "github:robcohen/nix-amd-npu";
  };

  outputs = { self, nixpkgs, nix-amd-npu, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        nix-amd-npu.nixosModules.default
        {
          hardware.amd-npu.enable = true;
        }
      ];
    };
  };
}
```

The module configures:
- Loads `amdxdna` kernel module
- Sets up udev rules for `/dev/accel*` devices
- Increases memlock limits (required for DMA buffer allocation)
- Installs XRT with XDNA plugin
- Sets `XILINX_XRT` environment variable

### Module Options

```nix
{
  hardware.amd-npu = {
    enable = true;
    # Optional: use a different package
    package = nix-amd-npu.packages.x86_64-linux.xrt-amdxdna;
  };
}
```

### Development Shell

```bash
nix develop github:robcohen/nix-amd-npu

# Check NPU detection
xrt-smi examine
```

### Manual Installation

If not using the NixOS module, you must manually configure memlock limits:

```nix
{
  environment.systemPackages = [
    nix-amd-npu.packages.x86_64-linux.xrt-amdxdna
  ];

  # Required for NPU buffer allocation
  security.pam.loginLimits = [
    { domain = "*"; type = "soft"; item = "memlock"; value = "unlimited"; }
    { domain = "*"; type = "hard"; item = "memlock"; value = "unlimited"; }
  ];

  environment.variables.XILINX_XRT = "${nix-amd-npu.packages.x86_64-linux.xrt-amdxdna}/opt/xilinx/xrt";
}
```

## Building from Source

```bash
git clone https://github.com/robcohen/nix-amd-npu
cd nix-amd-npu

# Build individual packages
nix build .#xrt
nix build .#xrt-plugin-amdxdna
nix build .#xrt-amdxdna  # combined (default)

# Check the flake
nix flake check
```

## Project Structure

```
nix-amd-npu/
├── flake.nix                      # Main flake (uses flake-parts)
├── parts/
│   ├── packages.nix               # Package definitions
│   ├── devshell.nix               # Development shell
│   └── nixos-module.nix           # NixOS module
└── pkgs/
    ├── xrt/default.nix            # XRT package
    └── xrt-plugin-amdxdna/default.nix  # XDNA plugin
```

## Troubleshooting

### mmap error: Resource temporarily unavailable

This occurs when the memlock limit is too low. The NPU driver needs to allocate 64MB+ DMA buffers.

**Solution:** Use the NixOS module, or manually set memlock limits and re-login:

```bash
# Check current limit
ulimit -l

# Should show "unlimited" after configuration
```

### No devices found

1. Check kernel module is loaded: `lsmod | grep amdxdna`
2. Check device exists: `ls -la /dev/accel*`
3. Check kernel messages: `sudo dmesg | grep amdxdna`

## Verified Hardware

- [ ] ThinkPad P16s Gen 4 AMD (Ryzen AI 7 PRO 350, Krackan NPU)
- [ ] Framework 16 (Ryzen AI 9 HX 370)
- [ ] Other Strix Point systems

## References

- [AMD xdna-driver](https://github.com/amd/xdna-driver)
- [Xilinx XRT](https://github.com/Xilinx/XRT)
- [Kernel amdxdna docs](https://docs.kernel.org/accel/amdxdna/amdnpu.html)
- [Arch Linux xrt-npu-git](https://aur.archlinux.org/packages/xrt-npu-git)

## License

Apache 2.0 (same as upstream XRT and xdna-driver)
