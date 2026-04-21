
import random

def make_inst(opcode, rd, rs1, rs2, imm):
    inst = 0
    inst |= (opcode & 0x3F) << 26
    inst |= (rd & 0xF) << 18
    inst |= (rs1 & 0xF) << 14
    inst |= (rs2 & 0xF) << 10
    inst |= (imm & 0x3FF)
    return inst

OP_NOP  = 0x00
OP_ADD  = 0x01
OP_SUB  = 0x02
OP_LDI  = 0x03
OP_AND  = 0x04
OP_OR   = 0x05
OP_MOV  = 0x06
OP_MUL  = 0x07
OP_LDW  = 0x10
OP_STW  = 0x11
OP_TID  = 0x12
OP_SMID = 0x13
OP_BEQ  = 0x20
OP_BNE  = 0x21
OP_JOIN = 0x22
OP_HALT = 0x3F

# Memory Layout
BASE_A = 1024
BASE_B = 1280 # 1024 + 64*4 (8x8 Matrix A)
BASE_C = 1408 # 1280 + 32*4 (8x4 Matrix B)

def generate_matmul_kernel():
    
    insts = []
    
    # --- 1. Calculate Row/Col from SMID/TID ---
    # R1 = SMID
    # R2 = TID
    insts.append(make_inst(OP_SMID, 1, 0, 0, 0))
    insts.append(make_inst(OP_TID,  2, 0, 0, 0))
    
    # GID (R3) = SMID * 8 + TID
    insts.append(make_inst(OP_MOV, 3, 1, 0, 0))   # R3 = SMID
    insts.append(make_inst(OP_ADD, 3, 3, 3, 0))   # R3 = SMID*2
    insts.append(make_inst(OP_ADD, 3, 3, 3, 0))   # R3 = SMID*4
    insts.append(make_inst(OP_ADD, 3, 3, 3, 0))   # R3 = SMID*8
    insts.append(make_inst(OP_ADD, 3, 3, 2, 0))   # R3 = GID

    # Col (R4) = GID % 4 -> GID & 3
    insts.append(make_inst(OP_LDI, 15, 0, 0, 3))  # R15 = 3
    insts.append(make_inst(OP_AND, 4, 3, 15, 0))  # R4 = Col

    # Recalculate Row: R5 = SMID * 2
    insts.append(make_inst(OP_MOV, 5, 1, 0, 0))
    insts.append(make_inst(OP_ADD, 5, 5, 5, 0)) # R5 = SMID*2
    
    # Add offset if TID >= 4. (Check bit 2 of TID)
    insts.append(make_inst(OP_LDI, 15, 0, 0, 4))
    insts.append(make_inst(OP_AND, 15, 2, 15, 0)) # R15 = 4 if TID>=4, else 0
    
    insts.append(make_inst(OP_LDI, 0, 0, 0, 0)) # Ensure R0 is 0
    
    # --- SIMT Branch ---
    # If (R15 == 0) -> Jump to SKIP
    # Else -> Fallthrough (Add 1)
    insts.append(make_inst(OP_BEQ, 15, 0, 2, 0)) 
    
    # Fallthrough Block (TID >= 4)
    insts.append(make_inst(OP_LDI, 15, 0, 0, 1)) # Load 1
    insts.append(make_inst(OP_ADD, 5, 5, 15, 0)) # Row += 1
    
    insts.append(make_inst(OP_JOIN, 0, 0, 0, 0)) # This is the target of BEQ (PC+2)
    
    # --- 2. Setup Loop ---
    # R6 = k (Loop Counter), init 0
    insts.append(make_inst(OP_LDI, 6, 0, 0, 0))
    
    # R7 = Accumulator, init 0
    insts.append(make_inst(OP_LDI, 7, 0, 0, 0))
    
    # Base Addrs
    insts.append(make_inst(OP_LDI, 10, 0, 0, BASE_A))
    insts.append(make_inst(OP_LDI, 11, 0, 0, BASE_B))
    insts.append(make_inst(OP_LDI, 12, 0, 0, BASE_C))
    
    # LOOP_START:
    idx_loop_start = len(insts)
    
    # Check k < 8. 
    insts.append(make_inst(OP_LDI, 15, 0, 0, 8))
    
    # Branch Exit: BEQ R6, R15, +Offset
    idx_loop_check = len(insts)
    insts.append(make_inst(OP_BEQ, 6, 15, 0, 0)) # Placeholder
    
    # Body:
    # Load A[Row][k]
    # Row * 8 = Row << 3
    insts.append(make_inst(OP_MOV, 8, 5, 0, 0)) # R8 = Row
    insts.append(make_inst(OP_ADD, 8, 8, 8, 0)) # *2
    insts.append(make_inst(OP_ADD, 8, 8, 8, 0)) # *4
    insts.append(make_inst(OP_ADD, 8, 8, 8, 0)) # *8
    insts.append(make_inst(OP_ADD, 8, 8, 6, 0)) # + k
    insts.append(make_inst(OP_ADD, 8, 8, 8, 0)) # *2
    insts.append(make_inst(OP_ADD, 8, 8, 8, 0)) # *4 (Bytes)
    insts.append(make_inst(OP_ADD, 8, 8, 10, 0)) # + BaseA
    insts.append(make_inst(OP_LDW, 8, 8, 0, 0))  # R8 = Val A
    
    # Load B[k][Col]
    # AddrB = BaseB + (k * 4 + Col) * 4
    insts.append(make_inst(OP_MOV, 9, 6, 0, 0)) # R9 = k
    insts.append(make_inst(OP_ADD, 9, 9, 9, 0)) # *2
    insts.append(make_inst(OP_ADD, 9, 9, 9, 0)) # *4
    insts.append(make_inst(OP_ADD, 9, 9, 4, 0)) # + Col
    insts.append(make_inst(OP_ADD, 9, 9, 9, 0)) # *2
    insts.append(make_inst(OP_ADD, 9, 9, 9, 0)) # *4 (Bytes)
    insts.append(make_inst(OP_ADD, 9, 9, 11, 0)) # + BaseB
    insts.append(make_inst(OP_LDW, 9, 9, 0, 0))  # R9 = Val B
    
    # MAC
    insts.append(make_inst(OP_MUL, 8, 8, 9, 0)) # R8 = A * B
    insts.append(make_inst(OP_ADD, 7, 7, 8, 0)) # Acc += Prod
    
    # Increment k
    insts.append(make_inst(OP_LDI, 15, 0, 0, 1))
    insts.append(make_inst(OP_ADD, 6, 6, 15, 0))
    
    # Jump back to START (Unconditional for active threads)
    # BEQ R0, R0
    jump_offset = idx_loop_start - len(insts)
    insts.append(make_inst(OP_BEQ, 0, 0, 0, jump_offset))
    
    # Target: JOIN instruction
    exit_offset = len(insts) - idx_loop_check
    insts[idx_loop_check] = make_inst(OP_BEQ, 6, 15, 0, exit_offset)
    
    insts.append(make_inst(OP_JOIN, 0, 0, 0, 0))
    
    # --- 3. Store Result ---
    # AddrC = BaseC + (Row * 4 + Col) * 4
    insts.append(make_inst(OP_MOV, 8, 5, 0, 0)) # Row
    insts.append(make_inst(OP_ADD, 8, 8, 8, 0)) # *2
    insts.append(make_inst(OP_ADD, 8, 8, 8, 0)) # *4
    insts.append(make_inst(OP_ADD, 8, 8, 4, 0)) # + Col
    insts.append(make_inst(OP_ADD, 8, 8, 8, 0)) # *2
    insts.append(make_inst(OP_ADD, 8, 8, 8, 0)) # *4
    insts.append(make_inst(OP_ADD, 8, 8, 12, 0)) # + BaseC
    insts.append(make_inst(OP_STW, 0, 8, 7, 0))  # Mem[R8] = Acc
    
    # Halt
    insts.append(make_inst(OP_HALT, 0, 0, 0, 0))
    
    return insts

def generate_data():
    # A: 8x8, B: 8x4
    random.seed(123)
    mat_a = [random.randint(0, 3) for _ in range(64)]
    mat_b = [random.randint(0, 3) for _ in range(32)]
    return mat_a, mat_b

def assemble():
    insts = generate_matmul_kernel()
    mat_a, mat_b = generate_data()
    
    with open("tests/program.hex", "w") as f:
        for inst in insts:
            b0 = (inst >> 0) & 0xFF
            b1 = (inst >> 8) & 0xFF
            b2 = (inst >> 16) & 0xFF
            b3 = (inst >> 24) & 0xFF
            f.write(f"{b0:02X}\n{b1:02X}\n{b2:02X}\n{b3:02X}\n")
            
        code_size = len(insts) * 4
        padding = BASE_A - code_size
        if padding < 0:
            print(f"Error: Code too large {code_size}")
            return
        
        for _ in range(padding):
            f.write("00\n")
            
        for val in mat_a:
            f.write(f"{(val)&0xFF:02X}\n00\n00\n00\n")
        for val in mat_b:
            f.write(f"{(val)&0xFF:02X}\n00\n00\n00\n")
            
    # Gold file for 8x4
    with open("tests/gold.txt", "w") as f:
        for r in range(8):
            f.write(f"Row {r}: ")
            for c in range(4):
                val = 0
                for k in range(8): # 8x8 * 8x4
                    a = mat_a[r*8 + k]
                    b = mat_b[k*4 + c]
                    val += a * b
                f.write(f"{val:4d} ")
            f.write("\n")

if __name__ == "__main__":
    assemble()
