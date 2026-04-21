import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tools.vibe_cuda as cuda

@cuda.jit(block_dim=32)
def load_test_kernel(a, c):
    gid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    gid = gid * bdim
    tid = cuda.threadIdx.x
    gid = gid + tid
    
    # Load from A
    val = a[gid]
    
    # Store to C
    c[gid] = val

def main():
    base_a = 1024
    base_c = 2560
    
    print(f"Compiling Load Test Kernel... A@{base_a}, C@{base_c}")
    
    asm = load_test_kernel(base_a, base_c)
    asm.base_data_addr = base_a
    asm.data_offset = 0
    
    # Fill A with 0..127
    vals_a = [i for i in range(128)]
    asm.data(vals_a)
    
    asm.write_hex("tests/program.hex")
    print("Written hex.")

if __name__ == "__main__":
    main()

