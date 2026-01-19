# Contributing to nix-amd-npu

Thank you for your interest in contributing! This guide will help you get started.

## Project Overview

This repository provides Nix packages and NixOS modules for AMD Ryzen AI NPU support. The goal is to enable hardware-accelerated AI inference on AMD Ryzen AI processors with a clean, reproducible Nix-based build system.

## Architecture

```
nix-amd-npu/
├── flake.nix              # Main entry point, defines overlay
├── lib/                   # Shared Nix functions
│   ├── default.nix        # Library entry point
│   ├── versions.nix       # Centralized version management
│   ├── meta.nix           # Shared meta attributes
│   └── vitis-common.nix   # Build helpers for Vitis AI packages
├── parts/                 # Flake-parts modules
│   ├── packages.nix       # Package definitions and tests
│   ├── devshell.nix       # Development shells
│   └── nixos-module.nix   # NixOS hardware module
└── pkgs/                  # Package derivations
    ├── xrt/               # Xilinx Runtime
    ├── xrt-plugin-amdxdna/# XDNA driver plugin
    ├── vitis-ai/          # Vitis AI components (8 packages)
    ├── onnxruntime-vitisai/ # ONNX Runtime with VitisAI EP
    ├── mlir-aie/          # MLIR-AIE for NPU kernels
    └── whisper-iron/      # Demo application
```

## Development Setup

### Prerequisites

- NixOS or Nix package manager with flakes enabled
- AMD Ryzen AI hardware (Strix Point, Hawk Point, or Krackan)
- Kernel 6.10+ (amdxdna driver in mainline 6.14+)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/nix-amd-npu
cd nix-amd-npu

# Enter the development shell
nix develop

# Verify NPU detection
xrt-smi examine
```

### Available Development Shells

| Shell | Purpose | Command |
|-------|---------|---------|
| `default` | Basic XRT development | `nix develop` |
| `vitisai` | ONNX Runtime with VitisAI EP | `nix develop .#vitisai` |
| `iron` | MLIR-AIE kernel development | `nix develop .#iron` |
| `iron-full` | Full IRON with pip support | `nix develop .#iron-full` |
| `whisper` | Whisper-IRON development | `nix develop .#whisper` |

## Building Packages

```bash
# Build the default package (xrt-amdxdna)
nix build

# Build a specific package
nix build .#xrt
nix build .#unilog
nix build .#onnxruntime-vitisai

# Run integration tests
nix flake check
```

## Adding a New Package

1. Create a new directory under `pkgs/<package-name>/`
2. Create `default.nix` with the derivation
3. Add the package to `flake.nix` overlay
4. Add the package to `parts/packages.nix`
5. Update `lib/versions.nix` with version info

### Package Template

```nix
{ lib
, stdenv
, fetchFromGitHub
, cmake
, ninja
, pkg-config
}:

stdenv.mkDerivation rec {
  pname = "my-package";
  version = "1.0.0";

  src = fetchFromGitHub {
    owner = "owner";
    repo = "repo";
    rev = "v${version}";
    hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
  };

  nativeBuildInputs = [ cmake ninja pkg-config ];
  buildInputs = [ ];

  meta = with lib; {
    description = "Description of my package";
    homepage = "https://github.com/owner/repo";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
```

### Common Build Fixes

The `lib/vitis-common.nix` provides helpers for common issues:

```nix
# GCC 15 compatibility (missing cstdint)
postPatch = lib.addGcc15Compat + ''
  # other patches...
'';

# Remove -Werror from cmake
postPatch = lib.removeWerror + ''...'';

# Create fake git repo for version detection
postPatch = lib.fakeGitRepo + ''...'';
```

## Version Management

All package versions are centralized in `lib/versions.nix`. When updating a package:

1. Update the version and hash in `lib/versions.nix`
2. Test the build: `nix build .#<package>`
3. Run checks: `nix flake check`

## Testing

### Integration Tests

Integration tests are defined in `parts/packages.nix`:

```bash
# Run all checks
nix flake check

# Run specific checks
nix build .#checks.x86_64-linux.xrt-binaries
nix build .#checks.x86_64-linux.plugin-library
```

### Manual Testing

```bash
# Verify XRT works
xrt-smi examine
xrt-smi validate

# Test ONNX Runtime VitisAI EP
nix develop .#vitisai
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## Code Style

- Use `nixfmt` for formatting Nix code
- Follow existing patterns in the codebase
- Add `meta` attributes to all packages
- Use `lib.licenses` for license specifications
- Prefer `substituteInPlace` over `sed` when possible

## Commit Messages

Follow conventional commits format:

```
feat: add new package foo
fix(xrt): resolve build issue with GCC 15
docs: update installation instructions
refactor(vitis-ai): extract common build helpers
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Run tests: `nix flake check`
5. Commit with descriptive message
6. Push and create a pull request

## Reporting Issues

When reporting issues, please include:

- NixOS version or Nix version
- Hardware details (CPU model)
- Kernel version: `uname -r`
- NPU status: `xrt-smi examine` output (if available)
- Relevant error messages

## License

This project is licensed under the MIT License. Contributions are accepted under the same license.

## Resources

- [XRT Documentation](https://xilinx.github.io/XRT/)
- [AMD Ryzen AI Developer Resources](https://www.amd.com/en/developer/resources/ryzen-ai-software.html)
- [MLIR-AIE Documentation](https://xilinx.github.io/mlir-aie/)
- [Nix Manual](https://nixos.org/manual/nix/stable/)
- [NixOS Wiki](https://wiki.nixos.org/)
