
import sys
import os
# Add project root to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tools.assembler import Assembler
import random

def main():
    asm = Assembler(base_data_addr=1024)
    
    # --- Data Generation ---
    # Matrix A: 8x8, Matrix B: 8x4
    random.seed(123)
    mat_a = [random.randint(0, 3) for _ in range(64)]
    mat_b = [random.randint(0, 3) for _ in range(32)]
    
    addr_a = asm.data(mat_a)
    addr_b = asm.data(mat_b)
    addr_c = 1024 + len(mat_a)*4 + len(mat_b)*4 # Address for Result C (8x4)
    # Note: Assembler.data only appends. We manually calculate C address or 
    # we could allocate space for C in data segment too (zeros).
    
    print(f"Data Layout: A@{addr_a}, B@{addr_b}, C@{addr_c}")

    # --- Kernel Code ---
    # R0: Zero
    # R1: SMID
    # R2: TID
    # R3: GID
    # R4: Col
    # R5: Row
    # R6: k (loop counter)
    # R7: Accumulator
    
    # 1. Calculate Coordinates
    asm.smid(1)
    asm.tid(2)
    
    # GID = SMID * 8 + TID
    asm.mov(3, 1)
    asm.add(3, 3, 3) # *2
    asm.add(3, 3, 3) # *4
    asm.add(3, 3, 3) # *8
    asm.add(3, 3, 2) # +TID -> GID
    
    # Col = GID % 4 (GID & 3)
    asm.ldi(15, 3)
    asm.and_(4, 3, 15)
    
    # Row Calculation (Complex mapping due to lack of DIV/SHIFT)
    # Row = SMID * 2 + (TID >= 4 ? 1 : 0)
    
    asm.mov(5, 1)
    asm.add(5, 5, 5) # R5 = SMID*2
    
    # Check TID >= 4
    asm.ldi(15, 4)
    asm.and_(15, 2, 15) # R15 != 0 if TID >= 4
    
    asm.ldi(0, 0) # R0 = 0
    asm.beq(15, 0, "SKIP_ROW_INC") # If TID < 4, Skip
    
    # Fallthrough (TID >= 4): Row++
    asm.ldi(15, 1)
    asm.add(5, 5, 15)
    
    asm.join() # Target for Divergence (PC+2 implicit target of BEQ if Taken needs to join?) 
               # Wait, BEQ jumps to Label. "SKIP_ROW_INC".
               # If Taken (TID < 4), jumps to SKIP_ROW_INC.
               # If Not Taken (TID >= 4), falls through, executes ADD, then hits SKIP_ROW_INC.
               # So we actually need a JOIN at SKIP_ROW_INC to handle the merge.
    
    asm.label("SKIP_ROW_INC")
    
    # 2. Loop Setup
    asm.ldi(6, 0) # k = 0
    asm.ldi(7, 0) # Acc = 0
    
    # Load Base Addresses
    asm.ldi(10, addr_a)
    asm.ldi(11, addr_b)
    asm.ldi(12, addr_c)
    
    asm.label("LOOP_START")
    
    # Loop Check: if k == 8, break
    asm.ldi(15, 8)
    asm.beq(6, 15, "LOOP_END")
    
    # 3. Loop Body
    # Load A[Row][k]
    # Addr = BaseA + (Row * 8 + k) * 4
    asm.mov(8, 5) # Row
    asm.add(8, 8, 8) # *2
    asm.add(8, 8, 8) # *4
    asm.add(8, 8, 8) # *8
    asm.add(8, 8, 6) # + k
    asm.add(8, 8, 8) # *2 (Byte offset)
    asm.add(8, 8, 8) # *4
    asm.add(8, 8, 10) # + BaseA
    asm.ldw(8, 8) # Val A
    
    # Load B[k][Col]
    # Addr = BaseB + (k * 4 + Col) * 4
    asm.mov(9, 6) # k
    asm.add(9, 9, 9) # *2
    asm.add(9, 9, 9) # *4
    asm.add(9, 9, 4) # + Col
    asm.add(9, 9, 9) # *2
    asm.add(9, 9, 9) # *4
    asm.add(9, 9, 11) # + BaseB
    asm.ldw(9, 9) # Val B
    
    # MAC
    asm.mul(8, 8, 9)
    asm.add(7, 7, 8)
    
    # Increment k
    asm.ldi(15, 1)
    asm.add(6, 6, 15)
    
    # Jump back
    asm.jump("LOOP_START")
    
    asm.label("LOOP_END")
    asm.join() # Re-converge loop exits
    
    # 4. Store Result
    # Addr = BaseC + (Row * 4 + Col) * 4
    asm.mov(8, 5) # Row
    asm.add(8, 8, 8) # *2
    asm.add(8, 8, 8) # *4
    asm.add(8, 8, 4) # + Col
    asm.add(8, 8, 8) # *2
    asm.add(8, 8, 8) # *4
    asm.add(8, 8, 12) # + BaseC
    asm.stw(8, 7) # Store Acc
    
    asm.halt()
    
    # --- Output ---
    asm.write_hex("tests/program.hex")
    
    # Generate Gold File for verification
    with open("tests/gold.txt", "w") as f:
        for r in range(8):
            f.write(f"Row {r}: ")
            for c in range(4):
                val = 0
                for k in range(8):
                    a = mat_a[r*8 + k]
                    b = mat_b[k*4 + c]
                    val += a * b
                f.write(f"{val:4d} ")
            f.write("\n")

if __name__ == "__main__":
    main()

