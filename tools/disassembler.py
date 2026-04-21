import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tools.isa import *

def disassemble(hex_file):
    with open(hex_file, 'r') as f:
        lines = f.read().splitlines()
    
    bytes_list = []
    for l in lines:
        if l.strip():
            bytes_list.append(int(l, 16))
            
    # Read 4 bytes as 1 inst
    pc = 0
    while pc < len(bytes_list):
        if pc + 4 > len(bytes_list): break
        
        inst = 0
        inst |= bytes_list[pc]
        inst |= bytes_list[pc+1] << 8
        inst |= bytes_list[pc+2] << 16
        inst |= bytes_list[pc+3] << 24
        
        opcode = (inst >> 26) & 0x3F
        rd = (inst >> 21) & 0x1F
        rs1 = (inst >> 16) & 0x1F
        rs2 = (inst >> 11) & 0x1F
        imm = inst & 0x7FF
        if imm & 0x400: imm -= 0x800 # Sign extend 11-bit
        
        op_name = "UNKNOWN"
        args = ""
        
        if opcode == OP_NOP: op_name = "NOP"
        elif opcode == OP_ADD: op_name = "ADD"; args = f"R{rd}, R{rs1}, R{rs2}"
        elif opcode == OP_SUB: op_name = "SUB"; args = f"R{rd}, R{rs1}, R{rs2}"
        elif opcode == OP_MUL: op_name = "MUL"; args = f"R{rd}, R{rs1}, R{rs2}"
        elif opcode == OP_LDI: op_name = "LDI"; args = f"R{rd}, {imm}"
        elif opcode == OP_BEQ: op_name = "BEQ"; args = f"R{rs1}, R{rs2}, {imm}"
        elif opcode == OP_BNE: op_name = "BNE"; args = f"R{rs1}, R{rs2}, {imm}"
        # elif opcode == OP_JUMP: op_name = "JUMP"; args = f"{imm}" # JUMP not in hardware, mapped to BEQ
        elif opcode == OP_LDW: op_name = "LDW"; args = f"R{rd}, {imm}(R{rs1})"
        elif opcode == OP_STW: op_name = "STW"; args = f"{imm}(R{rs1}), R{rs2}"
        elif opcode == OP_HALT: op_name = "HALT"
        elif opcode == OP_JOIN: op_name = "JOIN"
        elif opcode == OP_TID: op_name = "TID"; args = f"R{rd}"
        elif opcode == OP_WARPID: op_name = "WARPID"; args = f"R{rd}"
        elif opcode == OP_SMID: op_name = "SMID"; args = f"R{rd}"
        elif opcode == OP_SLL: op_name = "SLL"; args = f"R{rd}, R{rs1}, R{rs2}"
        elif opcode == OP_SRL: op_name = "SRL"; args = f"R{rd}, R{rs1}, R{rs2}"
        elif opcode == OP_AND: op_name = "AND"; args = f"R{rd}, R{rs1}, R{rs2}"
        elif opcode == OP_OR: op_name = "OR"; args = f"R{rd}, R{rs1}, R{rs2}"
        elif opcode == OP_MOV: op_name = "MOV"; args = f"R{rd}, R{rs1}"
        elif opcode == OP_CSR: op_name = "CSR"; args = f"R{rd}, {rs1}"
        elif opcode == OP_FADD: op_name = "FADD"; args = f"R{rd}, R{rs1}, R{rs2}"
        elif opcode == OP_FMUL: op_name = "FMUL"; args = f"R{rd}, R{rs1}, R{rs2}"
        elif opcode == OP_FADD4: op_name = "FADD4"; args = f"R{rd}, R{rs1}, R{rs2}"
        elif opcode == OP_FMUL4: op_name = "FMUL4"; args = f"R{rd}, R{rs1}, R{rs2}"
        
        print(f"{pc:04X}: {op_name} {args}")
        
        pc += 4
        if opcode == OP_HALT: break # Stop at HALT

if __name__ == "__main__":
    disassemble("tests/program.hex")

