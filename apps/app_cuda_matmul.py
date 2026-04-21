import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tools.vibe_cuda as cuda

M = 16
K = 16
N = 8

@cuda.jit(block_dim=32)
def matmul_kernel(a, b, c):
    # Grid: 128 threads (4 blocks of 32 threads)
    # C = A * B, where A is 16x16, B is 16x8, C is 16x8
    # Each thread calculates one element of C.
    
    # Calculate global ID
    gid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    gid = gid * bdim
    tid = cuda.threadIdx.x
    gid = gid + tid
    
    # Map 1D global_id to 2D (row, col) for C matrix (16x8)
    # WIDTH_B = 8
    row = gid >> 3 # gid / 8
    col = gid & 7  # gid % 8
    
    acc = 0
    
    # K loop (inner loop for matrix multiplication)
    for k in range(16): # K is 16
        # Load A[row][k]
        # idx_a = row * 16 + k
        tmp_a_idx = row << 4 
        tmp_a_idx = tmp_a_idx + k
        val_a = a[tmp_a_idx]

        # Load B[k][col]
        # idx_b = k * 8 + col
        tmp_b_idx = k << 3 
        tmp_b_idx = tmp_b_idx + col
        val_b = b[tmp_b_idx]

        # acc += val_a * val_b
        tmp_prod = val_a * val_b
        acc = acc + tmp_prod

    # Store C[row][col]
    c[gid] = acc

def main():
    # Use random data
    vals_a = [random.randint(0, 5) for _ in range(M * K)]
    vals_b = [random.randint(0, 5) for _ in range(K * N)]
    
    # Calculate Gold Result
    vals_c_gold = [0] * (M * N)
    for r in range(M):
        for c in range(N):
            acc = 0
            for k in range(K):
                acc += vals_a[r * K + k] * vals_b[k * N + c]
            vals_c_gold[r * N + c] = acc
            
    base_a = 1024
    base_b = base_a + (M * K) * 4 
    base_c = base_b + (K * N) * 4 
    
    print(f"Compiling MatMul Kernel... A@{base_a}, B@{base_b}, C@{base_c}")
    
    asm = matmul_kernel(base_a, base_b, base_c)
    
    asm.base_data_addr = base_a
    asm.data_offset = 0
    asm.data(vals_a)
    asm.data(vals_b)
    
    asm.write_hex("tests/program.hex")
    
    with open("tests/gold.txt", "w") as f:
        for r in range(32): 
            line = f"Row {r}:"
            for i in range(4):
                idx = r * 4
                if idx + i < len(vals_c_gold):
                    line += f" {vals_c_gold[idx+i]:4d} "
                else:
                    line += f" {0:4d} "
            f.write(line.strip() + " \n")

    print("Done! Hex written to tests/program.hex, Gold to tests/gold.txt")

if __name__ == "__main__":
    main()
