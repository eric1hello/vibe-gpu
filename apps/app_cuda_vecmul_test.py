import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tools.vibe_cuda as cuda

@cuda.jit(block_dim=32)
def vecmul_kernel(a, b, c):
    gid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    gid = gid * bdim
    tid = cuda.threadIdx.x
    gid = gid + tid
    
    val_a = a[gid]
    val_b = b[gid]
    prod = val_a * val_b
    c[gid] = prod

def main():
    base_a = 1024
    # Size A = 128 * 4 = 512.
    # Next avail = 1536.
    base_b = 1536
    base_c = 2560
    
    print(f"Compiling VecMul Kernel... A@{base_a}, B@{base_b}, C@{base_c}")
    
    asm = vecmul_kernel(base_a, base_b, base_c)
    asm.base_data_addr = base_a
    asm.data_offset = 0
    
    vals_a = [2 for i in range(128)]
    vals_b = [3 for i in range(128)]
    
    asm.data(vals_a)
    asm.data(vals_b)
    
    asm.write_hex("tests/program.hex")
    print("Written hex.")

if __name__ == "__main__":
    main()
