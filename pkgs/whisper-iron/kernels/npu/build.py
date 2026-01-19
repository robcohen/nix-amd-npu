#!/usr/bin/env python3
"""
Build script for NPU kernels.

Compiles C++ kernel sources to object files using Peano clang++ (from mlir_aie).
Must be run in the iron-fhs environment.

The Peano compiler is included in the mlir_aie pip package and provides
the LLVM-based compiler for AIE2 architecture.

Usage:
    python kernels/npu/build.py
"""

import os
import subprocess
import sys
from pathlib import Path

KERNEL_DIR = Path(__file__).parent

# C++ kernel sources and their output names
KERNELS = [
    ("add.cc", "add.cc.o"),
    ("gelu.cc", "gelu.cc.o"),
    ("softmax.cc", "softmax.cc.o"),
    ("layernorm.cc", "layernorm.cc.o"),
    ("matmul.cc", "matmul.cc.o"),
    ("conv1d.cc", "conv1d.cc.o"),
]

# Default compiler flags for AIE2 BF16 kernels
# From mlir-aie programming_examples
PEANOWRAP2_FLAGS = [
    "-O2",
    "-std=c++20",
    "--target=aie2-none-unknown-elf",
    "-DNDEBUG",
    "-I",  # Add AIE include path (will be filled in)
]


def find_peano_install():
    """Find Peano installation directory."""
    # Check PEANO_INSTALL_DIR env var first
    if "PEANO_INSTALL_DIR" in os.environ:
        peano_dir = Path(os.environ["PEANO_INSTALL_DIR"])
        if (peano_dir / "bin" / "clang++").exists():
            return peano_dir

    # Search in virtualenv site-packages for llvm-aie (note: hyphen not underscore)
    venv_path = os.environ.get("VIRTUAL_ENV", "")
    if venv_path:
        venv = Path(venv_path)
        for check_dir in [venv / "lib", venv / "lib64"]:
            # Try llvm-aie (hyphen) - this is how pip installs it
            for python_dir in check_dir.glob("python*/site-packages/llvm-aie"):
                if (python_dir / "bin" / "clang++").exists():
                    return python_dir
            # Also try llvm_aie (underscore) for older versions
            for python_dir in check_dir.glob("python*/site-packages/llvm_aie"):
                if (python_dir / "bin" / "clang++").exists():
                    return python_dir

    # Check if llvm-aie is installed via pip and we can find it via importlib
    try:
        import importlib.metadata
        dist = importlib.metadata.distribution("llvm-aie")
        for file in dist.files or []:
            if "bin/clang++" in str(file):
                # Get the full path
                llvm_dir = Path(dist.locate_file(file)).parent.parent
                if (llvm_dir / "bin" / "clang++").exists():
                    return llvm_dir
    except Exception:
        pass

    return None


def find_aie_include():
    """Find AIE include directory for aie_api headers."""
    try:
        import mlir_aie
        # mlir_aie.__file__ may be None for namespace packages
        if mlir_aie.__file__ is not None:
            aie_dir = Path(mlir_aie.__file__).parent
        else:
            # Try to find it via __path__
            if hasattr(mlir_aie, '__path__') and mlir_aie.__path__:
                aie_dir = Path(list(mlir_aie.__path__)[0])
            else:
                # Last resort: search in site-packages
                import site
                for sp in site.getsitepackages():
                    candidate = Path(sp) / "mlir_aie"
                    if candidate.exists():
                        aie_dir = candidate
                        break
                else:
                    return None

        # Headers are typically in include/ directory (for aie_api)
        for include_path in [
            aie_dir / "include",  # This contains aie_api/
            aie_dir / "aie_runtime_lib" / "AIE2",
        ]:
            if include_path.exists() and (include_path / "aie_api").exists():
                return include_path
            elif include_path.exists():
                # Keep as fallback
                return include_path
    except (ImportError, Exception):
        pass

    return None


def find_aie_tools():
    """Find AIE compilation tools."""
    tools = {}

    # Find Peano installation
    peano_dir = find_peano_install()
    if peano_dir:
        clang_path = peano_dir / "bin" / "clang++"
        if clang_path.exists():
            tools["peano_clang++"] = str(clang_path)
            tools["peano_dir"] = str(peano_dir)
            print(f"  Found Peano at: {peano_dir}")

    # Find AIE include directory
    aie_include = find_aie_include()
    if aie_include:
        tools["aie_include"] = str(aie_include)
        print(f"  Found AIE includes at: {aie_include}")

    # Find aiecc.py
    result = subprocess.run(["which", "aiecc.py"], capture_output=True, text=True)
    if result.returncode == 0:
        tools["aiecc"] = result.stdout.strip()

    # Fallback: try xchesscc (requires Vitis install)
    result = subprocess.run(["which", "xchesscc"], capture_output=True, text=True)
    if result.returncode == 0:
        tools["xchesscc"] = result.stdout.strip()

    return tools


def compile_kernel_peano(source: Path, output: Path, tools: dict) -> bool:
    """Compile kernel using Peano clang++ (recommended)."""
    clang = tools.get("peano_clang++")
    if not clang:
        return False

    # Build command with flags
    cmd = [clang]
    cmd.extend([
        "-O2",
        "-std=c++20",
        "--target=aie2-none-unknown-elf",
        "-DNDEBUG",
    ])

    # Add AIE include path if available
    aie_include = tools.get("aie_include")
    if aie_include:
        cmd.extend(["-I", aie_include])

    cmd.extend([
        "-c",
        str(source),
        "-o", str(output),
    ])

    print(f"\nCompiling {source.name} with Peano clang++...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAILED:")
        if result.stdout:
            print(f"    stdout: {result.stdout}")
        if result.stderr:
            print(f"    stderr: {result.stderr}")
        return False

    print(f"  OK: {output}")
    return True


def compile_kernel_xchesscc(source: Path, output: Path, tools: dict) -> bool:
    """Compile kernel using xchesscc (requires Vitis)."""
    xchesscc = tools.get("xchesscc")
    if not xchesscc:
        return False

    cmd = [
        xchesscc,
        "-p", "me",
        "-P", "aie2",
        "-c",
        "-o", str(output),
        str(source),
    ]

    print(f"\nCompiling {source.name} with xchesscc...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr}")
        return False

    print(f"  OK: {output}")
    return True


def main():
    print("=" * 60)
    print("  NPU Kernel Build")
    print("=" * 60)
    print("\nSearching for AIE tools...")

    tools = find_aie_tools()
    print(f"\nAvailable tools: {list(tools.keys())}")

    if not tools:
        print("\nERROR: No AIE compilation tools found.")
        print("Make sure you're in the iron-fhs environment with:")
        print("  pip install mlir_aie llvm-aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels")
        return 1

    success_count = 0
    fail_count = 0

    for source_name, output_name in KERNELS:
        source = KERNEL_DIR / source_name
        output = KERNEL_DIR / output_name

        if not source.exists():
            print(f"\nWARNING: Source not found: {source}")
            fail_count += 1
            continue

        # Try different compilers in order of preference
        compiled = False

        # 1. Try Peano clang++ (recommended, from llvm-aie pip package)
        if not compiled and "peano_clang++" in tools:
            compiled = compile_kernel_peano(source, output, tools)

        # 2. Try xchesscc (requires Vitis installation)
        if not compiled and "xchesscc" in tools:
            compiled = compile_kernel_xchesscc(source, output, tools)

        if compiled:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {success_count} compiled, {fail_count} failed")
    print("=" * 60)

    if fail_count > 0:
        print("\nTroubleshooting:")
        print("  1. Make sure llvm-aie is installed: pip install llvm-aie")
        print("  2. Check PEANO_INSTALL_DIR environment variable")
        print("  3. Verify AIE includes exist in mlir_aie package")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
