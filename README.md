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
    # Optional: change the group for device access and memlock limits (default: video)
    group = "video";
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

  boot.kernelModules = [ "amdxdna" ];

  # Allow video group to access NPU device
  services.udev.extraRules = ''
    SUBSYSTEM=="accel", KERNEL=="accel[0-9]*", GROUP="video", MODE="0660"
  '';

  # Required for NPU buffer allocation (only for video group members)
  security.pam.loginLimits = [
    { domain = "@video"; type = "soft"; item = "memlock"; value = "unlimited"; }
    { domain = "@video"; type = "hard"; item = "memlock"; value = "unlimited"; }
  ];

  environment.variables.XILINX_XRT = "${nix-amd-npu.packages.x86_64-linux.xrt-amdxdna}/opt/xilinx/xrt";
}
```

### Using the Overlay

For more control, you can apply the overlay directly:

```nix
{
  nixpkgs.overlays = [ nix-amd-npu.overlays.default ];

  # Then use pkgs.xrt, pkgs.xrt-plugin-amdxdna, or pkgs.xrt-amdxdna
  environment.systemPackages = [ pkgs.xrt-amdxdna ];
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
│   ├── packages.nix               # Package definitions + integration tests
│   ├── devshell.nix               # Development shell
│   └── nixos-module.nix           # NixOS module (nixpkgs-compatible)
└── pkgs/
    ├── xrt/default.nix            # XRT package (nixpkgs-style)
    └── xrt-plugin-amdxdna/default.nix  # XDNA plugin (nixpkgs-style)
```

## Outputs

| Output | Description |
|--------|-------------|
| `packages.x86_64-linux.xrt` | Xilinx Runtime |
| `packages.x86_64-linux.xrt-plugin-amdxdna` | AMD XDNA plugin |
| `packages.x86_64-linux.xrt-amdxdna` | Combined package (default) |
| `overlays.default` | Nixpkgs overlay adding `pkgs.xrt`, `pkgs.xrt-plugin-amdxdna`, `pkgs.xrt-amdxdna` |
| `nixosModules.amd-npu` | NixOS hardware module |
| `checks.x86_64-linux.*` | Integration tests |

## Contributing to nixpkgs

This flake is structured for easy upstreaming to nixpkgs:

1. **Packages** in `pkgs/` are standalone nixpkgs-style derivations
2. **Module** in `parts/nixos-module.nix` uses `pkgs.*` with flake fallback
3. **No flake-specific code** in package definitions

To upstream:
```
pkgs/xrt/default.nix              → pkgs/by-name/xr/xrt/package.nix
pkgs/xrt-plugin-amdxdna/          → pkgs/by-name/xr/xrt-plugin-amdxdna/package.nix
parts/nixos-module.nix            → nixos/modules/hardware/amd-npu.nix
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
