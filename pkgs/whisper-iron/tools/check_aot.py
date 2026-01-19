#!/usr/bin/env python3
"""Check AOT compilation options in IRON."""
from aie.iron import Program
import inspect

print("Program class methods:")
for m in dir(Program):
    if not m.startswith('_'):
        print(f"  {m}")

# Check if there's a save/load method
print("\nChecking for AOT-related methods:")
if hasattr(Program, 'save'):
    print("  Program.save: EXISTS")
if hasattr(Program, 'load'):
    print("  Program.load: EXISTS")
if hasattr(Program, 'compile'):
    print("  Program.compile: EXISTS")
if hasattr(Program, 'to_xclbin'):
    print("  Program.to_xclbin: EXISTS")

# Check resolve_program output
print("\nProgram.resolve_program signature:")
print(f"  {inspect.signature(Program.resolve_program)}")

# Check if there's a way to get the compiled artifact
print("\nProgram attributes:")
p = Program.__init__
print(f"  __init__ signature: {inspect.signature(p)}")
