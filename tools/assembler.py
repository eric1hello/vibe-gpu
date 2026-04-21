from tools.isa import *
import os

class Assembler:
    def __init__(self, base_data_addr=1024):
        self.insts = []
        self.labels = {} # name -> index
        self.patches = [] # (index, label_name, opcode, rd, rs1, rs2)
        self.data_seg = [] # list of bytes/words
        self.base_data_addr = base_data_addr
        self.data_offset = 0
        
    def get_current_index(self):
        return len(self.insts)

    # --- Instructions ---
    def nop(self):
        self.insts.append(make_inst(OP_NOP, 0, 0, 0, 0))

    def add(self, rd, rs1, rs2):
        self.insts.append(make_inst(OP_ADD, rd, rs1, rs2, 0))

    def sub(self, rd, rs1, rs2):
        self.insts.append(make_inst(OP_SUB, rd, rs1, rs2, 0))

    def mul(self, rd, rs1, rs2):
        self.insts.append(make_inst(OP_MUL, rd, rs1, rs2, 0))

    def sll(self, rd, rs1, rs2):
        self.insts.append(make_inst(OP_SLL, rd, rs1, rs2, 0))

    def srl(self, rd, rs1, rs2):
        self.insts.append(make_inst(OP_SRL, rd, rs1, rs2, 0))

    def fadd(self, rd, rs1, rs2):
        self.insts.append(make_inst(OP_FADD, rd, rs1, rs2, 0))

    def fmul(self, rd, rs1, rs2):
        self.insts.append(make_inst(OP_FMUL, rd, rs1, rs2, 0))

    def fadd4(self, rd, rs1, rs2):
        """FP4 E2M1 浮点加（使用寄存器低 4 位）。"""
        self.insts.append(make_inst(OP_FADD4, rd, rs1, rs2, 0))

    def fmul4(self, rd, rs1, rs2):
        """FP4 E2M1 浮点乘（使用寄存器低 4 位）。"""
        self.insts.append(make_inst(OP_FMUL4, rd, rs1, rs2, 0))

    def csr(self, rd, rs1):
        # Read CSR: rs1 is CSR index, rd is destination
        self.insts.append(make_inst(OP_CSR, rd, rs1, 0, 0))

    def ldi(self, rd, imm):
        self.insts.append(make_inst(OP_LDI, rd, 0, 0, imm))

    def and_(self, rd, rs1, rs2):
        self.insts.append(make_inst(OP_AND, rd, rs1, rs2, 0))

    def or_(self, rd, rs1, rs2):
        self.insts.append(make_inst(OP_OR, rd, rs1, rs2, 0))

    def mov(self, rd, rs1):
        self.insts.append(make_inst(OP_MOV, rd, rs1, 0, 0))

    def ldw(self, rd, rs1):
        # Load Word from address in rs1
        self.insts.append(make_inst(OP_LDW, rd, rs1, 0, 0))

    def stw(self, rs1, rs2):
        # Store Word rs2 to address in rs1
        self.insts.append(make_inst(OP_STW, 0, rs1, rs2, 0))

    def tid(self, rd):
        self.insts.append(make_inst(OP_TID, rd, 0, 0, 0))

    def smid(self, rd):
        self.insts.append(make_inst(OP_SMID, rd, 0, 0, 0))
    
    def warpid(self, rd):
        self.insts.append(make_inst(OP_WARPID, rd, 0, 0, 0))

    def halt(self):
        self.insts.append(make_inst(OP_HALT, 0, 0, 0, 0))
        
    def join(self):
        self.insts.append(make_inst(OP_JOIN, 0, 0, 0, 0))

    # --- Control Flow ---
    def label(self, name):
        self.labels[name] = len(self.insts)

    def beq(self, rs1, rs2, label):
        self.patches.append((len(self.insts), label, OP_BEQ, 0, rs1, rs2))
        self.insts.append(0) # Placeholder

    def bne(self, rs1, rs2, label):
        self.patches.append((len(self.insts), label, OP_BNE, 0, rs1, rs2))
        self.insts.append(0) # Placeholder
        
    def jump(self, label):
        # Unconditional Jump using BEQ R0, R0
        self.patches.append((len(self.insts), label, OP_BEQ, 0, 0, 0))
        self.insts.append(0)

    # --- Data Management ---
    def data(self, values):
        """
        Append values (list of ints) to data segment and return the starting address.
        Values are 32-bit integers.
        """
        addr = self.base_data_addr + self.data_offset
        for v in values:
            # Store as 4 bytes (Little Endian)
            self.data_seg.append(v & 0xFF)
            self.data_seg.append((v >> 8) & 0xFF)
            self.data_seg.append((v >> 16) & 0xFF)
            self.data_seg.append((v >> 24) & 0xFF)
            self.data_offset += 4
        return addr

    def write_hex(self, filename="tests/program.hex"):
        # Resolve patches
        for idx, label, op, rd, rs1, rs2 in self.patches:
            if label not in self.labels:
                raise ValueError(f"Undefined label: {label}")
            target = self.labels[label]
            offset = target - idx
            self.insts[idx] = make_inst(op, rd, rs1, rs2, offset)

        with open(filename, "w") as f:
            # Write Code
            for inst in self.insts:
                b0 = (inst >> 0) & 0xFF
                b1 = (inst >> 8) & 0xFF
                b2 = (inst >> 16) & 0xFF
                b3 = (inst >> 24) & 0xFF
                f.write(f"{b0:02X}\n{b1:02X}\n{b2:02X}\n{b3:02X}\n")
            
            # Padding
            code_size = len(self.insts) * 4
            padding = self.base_data_addr - code_size
            if padding < 0:
                raise ValueError(f"Code too large ({code_size} bytes) for data start ({self.base_data_addr})")
            
            for _ in range(padding):
                f.write("00\n")
                
            # Write Data
            for b in self.data_seg:
                f.write(f"{b:02X}\n")
        
        print(f"Successfully wrote {filename}")
