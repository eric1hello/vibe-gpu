import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tools.vibe_cuda as cuda

@cuda.jit(block_dim=32)
def debug_kernel(c):
    gid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    gid = gid * bdim
    tid = cuda.threadIdx.x
    gid = gid + tid
    
    c[gid] = gid

def main():
    # C array of size 128
    base_c = 2560
    
    print(f"Compiling Debug Kernel... C@{base_c}")
    
    asm = debug_kernel(base_c)
    asm.base_data_addr = base_c
    asm.data_offset = 0
    # No input data, just output buffer
    # We need to pad data segment so header is correct?
    # logic says "Code too large for data start". 
    # If code is small, we pad.
    
    # Initialize C with 0xFF in hex file to ensure we overwrite
    vals_c = [0xFF] * 128
    asm.data(vals_c)
    
    asm.write_hex("tests/program.hex")
    
    # Verify manually for now
    print("Written hex.")

if __name__ == "__main__":
    main()

