`include "defines.svh"
// 算术逻辑单元：
//   - CUDA Core 类路径：整数、FP8、标量 FP4（FADD4/FMUL4，每线程对两个半字节标量运算）
//   - Tensor Core 类路径：tensor_core（TCDP4，对寄存器低 16 位内 4 路 FP4 做点积规约）
module alu (
    input logic [5:0] opcode,
    input logic [`DATA_WIDTH-1:0] op_a,
    input logic [`DATA_WIDTH-1:0] op_b,
    input logic [`DATA_WIDTH-1:0] imm,
    input logic [2:0] thread_id, // Used for TID instruction (3 bits for 8 threads)
    input logic [`WARP_ID_WIDTH-1:0] warp_id, // New: Warp ID
    input logic [1:0] sm_id,     // Used for SMID instruction
    
    // CSR Inputs
    input logic [`DATA_WIDTH-1:0] csr_cycle,
    input logic [`DATA_WIDTH-1:0] csr_instret,
    
    output logic [`DATA_WIDTH-1:0] result
);

    logic [7:0] fp8_add_res;
    logic [7:0] fp8_mul_res;
    logic [3:0] fp4_add_res;
    logic [3:0] fp4_mul_res;
    logic [3:0] tc_dp4_res;

    // FP8：使用操作数低 8 位
    fp8_unit u_fp8_unit (
        .a(op_a[7:0]),
        .b(op_b[7:0]),
        .add_res(fp8_add_res),
        .mul_res(fp8_mul_res)
    );

    // FP4（演进）：使用操作数低 4 位，结果零扩展到 32 位
    fp4_unit u_fp4_unit (
        .a(op_a[3:0]),
        .b(op_b[3:0]),
        .add_res(fp4_add_res),
        .mul_res(fp4_mul_res)
    );

    // Tensor Core：整字低 16 位为 4×FP4 lane，做点积
    tensor_core u_tensor_core (
        .op_a(op_a),
        .op_b(op_b),
        .dp4_res(tc_dp4_res)
    );

    always_comb begin
        case (opcode)
            `OP_ADD:  result = op_a + op_b;
            `OP_SUB:  result = op_a - op_b;
            `OP_LDI:  result = imm; // Immediate is already sign extended in decode
            `OP_AND:  result = op_a & op_b;
            `OP_OR:   result = op_a | op_b;
            `OP_MOV:  result = op_a;
            `OP_MUL:  result = op_a * op_b;
            `OP_SLL:  result = op_a << op_b[4:0]; // Shift Left
            `OP_SRL:  result = op_a >> op_b[4:0]; // Shift Right Logical
            `OP_FADD: result = {24'b0, fp8_add_res};
            `OP_FMUL: result = {24'b0, fp8_mul_res};
            `OP_FADD4: result = {28'b0, fp4_add_res};
            `OP_FMUL4: result = {28'b0, fp4_mul_res};
            `OP_TCDP4: result = {28'b0, tc_dp4_res};
            `OP_TID:  result = {29'b0, thread_id};
            `OP_SMID: result = {30'b0, sm_id};
            `OP_WARPID: result = {30'b0, warp_id};
            `OP_CSR: begin
                if (op_a == 0) result = csr_cycle;
                else if (op_a == 1) result = csr_instret;
                else result = 0;
            end
            // Address Calculation for Memory Ops (Reg + Imm)
            `OP_LDW:  result = op_a + imm; 
            `OP_STW:  result = op_a + imm; 
            default:  result = 32'b0;
        endcase
    end

endmodule
