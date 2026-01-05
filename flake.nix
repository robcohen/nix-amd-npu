{
  description = "AMD Ryzen AI NPU support for NixOS (XRT + XDNA driver)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, nixpkgs, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      imports = [
        ./parts/packages.nix
        ./parts/devshell.nix
        ./parts/nixos-module.nix
      ];

      perSystem = { system, ... }: {
        # Allow unfree packages (XRT has some unfree components)
        _module.args.pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      };
    };
}
