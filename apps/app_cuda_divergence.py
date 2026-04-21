import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tools.vibe_cuda as cuda

@cuda.jit(block_dim=32)
def divergence_kernel(c):
    gid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    gid = gid * bdim
    tid = cuda.threadIdx.x
    gid = gid + tid
    
    # Check if odd/even
    # mask = 1
    # is_odd = gid & mask
    mask = 1
    is_odd = gid & mask
    
    # If is_odd != 0 (True)
    if is_odd:
        c[gid] = 1
    else:
        c[gid] = 2
        
    # Reconverge
    c[gid] = c[gid] + 10

def main():
    base_c = 2560
    print(f"Compiling Divergence Kernel... C@{base_c}")
    
    asm = divergence_kernel(base_c)
    asm.base_data_addr = 1024
    asm.data_offset = 0
    
    # Padding
    asm.data([0] * 384)
    
    asm.write_hex("tests/program.hex")

if __name__ == "__main__":
    main()

