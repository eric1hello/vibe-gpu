
import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tools.vibe_cuda as cuda

# Define CUDA Kernel
# Configure Block Size = 32
# Hardware has 128 threads. So we expect Grid Dim = 4.
@cuda.jit(block_dim=32)
def vec_add_kernel(a, b, c):
    # 1. Get Coordinates
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    
    # 2. Calculate Global Index
    # i = bid * bdim + tid
    idx = bid * bdim + tid
    
    # 3. Do Computation
    val_a = a[idx]
    val_b = b[idx]
    res = val_a + val_b
    
    # 4. Store Result
    c[idx] = res

def main():
    # 128 elements (Total HW Threads)
    N = 128
    
    # Data
    vals_a = [i for i in range(N)]
    vals_b = [2 * i for i in range(N)]
    vals_c_gold = [a + b for a, b in zip(vals_a, vals_b)]
    
    base_a = 1024
    base_b = base_a + N * 4
    base_c = base_b + N * 4
    
    print(f"Compiling CUDA Kernel... A@{base_a}, B@{base_b}, C@{base_c}")
    
    # Compile & Link
    asm = vec_add_kernel(base_a, base_b, base_c)
    
    # Inject Data
    asm.base_data_addr = base_a
    asm.data_offset = 0
    asm.data(vals_a)
    asm.data(vals_b)
    
    asm.write_hex("tests/program.hex")
    
    # Verify Gold
    with open("tests/gold.txt", "w") as f:
        # Format matches gpu_top dump: 4 words per row
        for r in range(N // 4): 
            idx_start = r * 4
            line = f"Row {r}:"
            for k in range(4):
                idx = idx_start + k
                line += f" {vals_c_gold[idx]:4d} "
            f.write(line.strip() + " \n")
            
    print("Done! Run 'make run' in sim/ directory.")

if __name__ == "__main__":
    main()
