import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tools.vibe_cuda as cuda

# Using simple inline assembly for FP8 until CUDA frontend supports intrinsics
# Or just map it manually.
# Actually, let's stick to the Assembler level for this specific hardware test
# since it's testing a specific hardware unit (FP8) which might not have high-level language support yet.
# Re-writing using Assembler.

from tools.assembler import Assembler

def main():
    # FP8 E4M3 Values (Bias 7)
    vals_a = [0x38, 0x3C, 0x38, 0x40] * 8 # 32 elements
    vals_b = [0x38, 0x40, 0x30, 0x40] * 8
    
    expected_add = [0x40, 0x46, 0x3C, 0x48] * 8
    expected_mul = [0x38, 0x44, 0x30, 0x48] * 8
    
    base_a = 1024
    base_b = 1152
    base_c = 1408
    
    print(f"Compiling FP8 Kernel (Assembler)... A@{base_a}, B@{base_b}, C@{base_c}")
    
    asm = Assembler(base_data_addr=base_a)
    
    # R1 = PID
    asm.program_id(1) 
    
    # Offset = PID * 4 (4 bytes per word? No, we are loading packed bytes? 
    # Wait, the original test was loading Words. 
    # "val_a = triton.load(a_ptr + offset_in)"
    # Let's assume 1 word per thread for simplicity in this port.
    
    # R2 = Base A
    asm.ldi(2, 1024) # Base A
    # R3 = Base B
    asm.ldi(3, 1152) # Base B
    # R4 = Base C
    asm.ldi(4, 1408) # Base C
    
    # R5 = Offset (PID * 4)
    asm.sll(5, 1, 2) # PID << 2
    
    # R6 = Addr A = Base A + Offset
    asm.add(6, 2, 5)
    # R7 = Val A
    asm.ldw(7, 6)
    
    # R8 = Addr B = Base B + Offset
    asm.add(8, 3, 5)
    # R9 = Val B
    asm.ldw(9, 8)
    
    # FP8 Ops
    # R10 = Add Result
    asm.fadd(10, 7, 9)
    # R11 = Mul Result
    asm.fmul(11, 7, 9)
    
    # Store
    # Output Offset = PID * 8 (Store 2 results)
    # R12 = PID * 8
    asm.sll(12, 1, 3) 
    
    # Addr C_Add = Base C + Offset
    asm.add(13, 4, 12)
    asm.stw(13, 10)
    
    # Addr C_Mul = Base C + Offset + 4
    asm.ldi(14, 4)
    asm.add(15, 13, 14)
    asm.stw(15, 11)
    
    asm.halt()
    
    # Helper to inject data
    # We need to extend Assembler to support program_id macro? 
    # No, I implemented it in CudaCompiler but not Assembler class directly.
    # Actually Assembler has `tid`, `warpid`, `smid`. 
    # Let's implement `program_id` logic manually using asm instructions.
    
    # RE-DOING ASM GENERATION WITH CORRECT HARDWARE ID LOGIC
    asm = Assembler(base_data_addr=base_a)
    
    # --- Program ID Calculation ---
    # HW_ID = SMID * 32 + WARPID * 8 + TID
    
    # R1 = HW_ID
    asm.warpid(1)
    asm.ldi(2, 3)
    asm.sll(1, 1, 2) # WarpID * 8
    
    asm.tid(2)
    asm.add(1, 1, 2) # + TID
    
    asm.smid(2)
    asm.ldi(3, 5)
    asm.sll(2, 2, 3) # SMID * 32
    asm.add(1, 1, 2) # Final HW_ID in R1
    
    # --- Load/Store Logic ---
    # R2 = Base A (1024)
    asm.ldi(2, 1024)
    # R3 = Offset = HW_ID * 4
    asm.ldi(4, 2)
    asm.sll(3, 1, 4) 
    
    # R4 = Addr A
    asm.add(4, 2, 3)
    # R5 = Val A
    asm.ldw(5, 4)
    
    # R6 = Base B (1152)
    asm.ldi(6, 1152)
    # R7 = Addr B
    asm.add(7, 6, 3)
    # R8 = Val B
    asm.ldw(8, 7)
    
    # --- FP8 Ops ---
    asm.fadd(9, 5, 8)
    asm.fmul(10, 5, 8)
    
    # --- Store ---
    # R11 = Base C (1408)
    asm.ldi(11, 1408)
    # R12 = Out Offset = HW_ID * 8
    asm.ldi(13, 3)
    asm.sll(12, 1, 13)
    
    # R14 = Addr C_Add
    asm.add(14, 11, 12)
    asm.stw(14, 9)
    
    # R15 = Addr C_Mul = Addr C_Add + 4
    asm.ldi(16, 4)
    asm.add(15, 14, 16)
    asm.stw(15, 10)
    
    asm.halt()
    
    # Inject Data
    asm.base_data_addr = base_a
    asm.data_offset = 0
    asm.data(vals_a)
    asm.data(vals_b)
    
    asm.write_hex("tests/program.hex")
    
    with open("tests/gold.txt", "w") as f:
        for r in range(8):
            idx_start = r * 2 
            t0 = idx_start
            t1 = idx_start + 1
            line = f"Row {r}:   {expected_add[t0]:2d}   {expected_mul[t0]:2d}   {expected_add[t1]:2d}   {expected_mul[t1]:2d} \n"
            f.write(line)
            
    print("Done! FP8 Test compiled.")

if __name__ == "__main__":
    main()
