import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tools.vibe_cuda as cuda

@cuda.jit(block_dim=32)
def loop_test_kernel(c):
    gid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    gid = gid * bdim
    tid = cuda.threadIdx.x
    gid = gid + tid
    
    acc = 0
    for i in range(16):
        acc = acc + 1
    
    c[gid] = acc

def main():
    base_c = 2560
    
    print(f"Compiling Loop Test Kernel... C@{base_c}")
    
    asm = loop_test_kernel(base_c)
    asm.base_data_addr = 1024
    asm.data_offset = 0
    
    # Padding to reach C
    # 2560 - 1024 = 1536 bytes.
    # 1536 / 4 = 384 words.
    asm.data([0] * 384)
    
    asm.write_hex("tests/program.hex")
    print("Written hex.")

if __name__ == "__main__":
    main()

