
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tools.assembler import Assembler
import random

def main():
    asm = Assembler(base_data_addr=512)
    
    # --- Data ---
    # Vector A: 32 elements
    # Vector B: 32 elements
    # Result C: 32 elements
    # Grid: 32 threads (4 SMs * 8 Threads). Perfect mapping.
    
    N = 32
    random.seed(999)
    vec_a = [random.randint(0, 10) for _ in range(N)]
    vec_b = [random.randint(0, 10) for _ in range(N)]
    
    addr_a = asm.data(vec_a)
    addr_b = asm.data(vec_b)
    addr_c = 512 + N*4 + N*4
    
    print(f"Vector Mul: N={N}. A@{addr_a}, B@{addr_b}, C@{addr_c}")
    
    # --- Kernel ---
    # R1: GID
    # R2: Addr
    # R3: Val A
    # R4: Val B
    # R5: Product
    
    # 1. Get GID = SMID * 8 + TID
    asm.smid(1)
    asm.tid(2)
    
    asm.mov(1, 1)
    asm.add(1, 1, 1) # *2
    asm.add(1, 1, 1) # *4
    asm.add(1, 1, 1) # *8
    asm.add(1, 1, 2) # +TID
    
    # 2. Load A[GID]
    # Addr = BaseA + GID * 4
    asm.ldi(10, addr_a)
    asm.mov(2, 1)
    asm.add(2, 2, 2) # *2
    asm.add(2, 2, 2) # *4 (Bytes)
    asm.add(2, 2, 10) # +Base
    asm.ldw(3, 2)
    
    # 3. Load B[GID]
    asm.ldi(11, addr_b)
    asm.mov(2, 1)
    asm.add(2, 2, 2)
    asm.add(2, 2, 2)
    asm.add(2, 2, 11)
    asm.ldw(4, 2)
    
    # 4. Mul
    asm.mul(5, 3, 4)
    
    # 5. Store C[GID]
    asm.ldi(12, addr_c)
    asm.mov(2, 1)
    asm.add(2, 2, 2)
    asm.add(2, 2, 2)
    asm.add(2, 2, 12)
    asm.stw(2, 5)
    
    asm.halt()
    
    asm.write_hex("tests/program.hex")
    
    # Gold
    # Note: RTL gpu_top.sv verification is hardcoded for MatMul addresses/layout currently.
    # We might need to update gpu_top.sv to be generic or just manually check simulation trace.
    # Or update gpu_top.sv to not verify if not running matmul?
    # Let's just generate the gold text for reference.
    with open("tests/gold.txt", "w") as f:
        for i in range(N):
            val = vec_a[i] * vec_b[i]
            f.write(f"Idx {i}: {val}\n")

if __name__ == "__main__":
    main()

