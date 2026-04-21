`ifndef DEFINES_SVH
`define DEFINES_SVH

// Architecture Parameters
`define DATA_WIDTH 32
`define ADDR_WIDTH 32
`define REG_ADDR_WIDTH 5   // Increased to 5 bits
`define NUM_REGS 32        // Increased to 32 registers
`define THREADS_PER_WARP 8 // Threads per Warp (Hardware Threads)
`define WARPS_PER_SM 4
`define WARP_ID_WIDTH 2
`define SM_COUNT 4
`define IMEM_SIZE 2048
`define DMEM_SIZE 4096

// Opcodes
`define OP_NOP  6'h00
`define OP_ADD  6'h01
`define OP_SUB  6'h02
`define OP_LDI  6'h03
`define OP_AND  6'h04
`define OP_OR   6'h05
`define OP_MOV  6'h06
`define OP_MUL  6'h07
`define OP_FADD 6'h08
`define OP_FMUL 6'h09
`define OP_CSR  6'h0A
`define OP_SLL  6'h0B
`define OP_SRL  6'h0C
`define OP_FADD4 6'h0D  // FP4 E2M1 浮点加（操作数为寄存器低 4 位）
`define OP_FMUL4 6'h0E  // FP4 E2M1 浮点乘（CUDA Core 类标量）
`define OP_TCDP4 6'h0F  // Tensor Core 类：4×FP4 点积（低 16 位打包）
`define OP_LDW  6'h10
`define OP_STW  6'h11
`define OP_TID  6'h12
`define OP_SMID 6'h13
`define OP_WARPID 6'h14
`define OP_BEQ  6'h20
`define OP_BNE  6'h21
`define OP_JOIN 6'h22
`define OP_HALT 6'h3F

// Cache Parameters
`define CACHE_LINE_SIZE 16
`define CACHE_SIZE 256
`define TAG_WIDTH 24
`define INDEX_WIDTH 4

// SIMT Stack
`define STACK_DEPTH 8

`endif
