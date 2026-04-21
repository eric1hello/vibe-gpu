
# Instruction Encodings
# New Layout:
# [31:26] Opcode (6)
# [25:21] RD (5)
# [20:16] RS1 (5)
# [15:11] RS2 (5)
# [10:0]  Imm (11)

def make_inst(opcode, rd, rs1, rs2, imm):
    inst = 0
    inst |= (opcode & 0x3F) << 26
    inst |= (rd & 0x1F) << 21
    inst |= (rs1 & 0x1F) << 16
    inst |= (rs2 & 0x1F) << 11
    inst |= (imm & 0x7FF)
    return inst

# Opcodes
OP_NOP  = 0x00
OP_ADD  = 0x01
OP_SUB  = 0x02
OP_LDI  = 0x03
OP_AND  = 0x04
OP_OR   = 0x05
OP_MOV  = 0x06
OP_MUL  = 0x07
OP_FADD = 0x08
OP_FMUL = 0x09
OP_CSR  = 0x0A
OP_SLL  = 0x0B
OP_SRL  = 0x0C
OP_FADD4 = 0x0D
OP_FMUL4 = 0x0E
OP_TCDP4 = 0x0F
OP_LDW  = 0x10
OP_STW  = 0x11
OP_TID  = 0x12
OP_SMID = 0x13
OP_WARPID = 0x14
OP_BEQ  = 0x20
OP_BNE  = 0x21
OP_JOIN = 0x22
OP_HALT = 0x3F
