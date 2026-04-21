// ============================================================================
// Tensor Core（简化模型）
// 与「CUDA Core 类」标量 FP4（FADD4/FMUL4，见 fp4_unit）不同，本模块实现
// **向量片段规约**：对两个寄存器中各 **4 个 FP4 半字节**（低 16 位，lane0..3）做点积。
// 语义类比 Tensor Core 对矩阵片段的乘加规约（WMMA 极简子集）。
// 数值约定与 fp4_unit / tools/fp4_soft.py 一致（FIX=256）。
// ============================================================================

module tensor_core (
    input  logic [31:0] op_a,
    input  logic [31:0] op_b,
    output logic [3:0]  dp4_res
);

    localparam int unsigned FIX = 256;

    function automatic logic signed [31:0] decode_fp4_lane(input logic [3:0] f);
        logic s;
        logic [1:0] e;
        logic m;
        logic signed [31:0] mag;
        begin
            if (f[2:1] == 2'b00)
                decode_fp4_lane = 0;
            else begin
                s = f[3];
                e = f[2:1];
                m = f[0];
                mag = ($signed(32'(32'd1 << (e - 1))) * $signed(32'(32'd2 + {31'b0, m})) * FIX) >>> 1;
                decode_fp4_lane = s ? -mag : mag;
            end
        end
    endfunction

    function automatic logic [3:0] encode_fp4_val(input logic signed [31:0] val_fix);
        logic signed [31:0] absv;
        logic [1:0] best_e;
        logic best_m;
        integer best_err, err, t;
        begin
            if (val_fix == 0) return 4'b0000;
            absv = val_fix < 0 ? -val_fix : val_fix;
            best_err = 2147483647;
            best_e = 2'd1;
            best_m = 1'b0;
            for (int e = 1; e <= 3; e++) begin
                for (int m = 0; m <= 1; m++) begin
                    t = ((32'd1 << (e - 1)) * (32'd2 + m) * FIX) >>> 1;
                    err = (absv > t) ? (absv - t) : (t - absv);
                    if (err < best_err) begin
                        best_err = err;
                        best_e = e[1:0];
                        best_m = m[0];
                    end
                end
            end
            encode_fp4_val = {val_fix < 0, best_e[1:0], best_m};
        end
    endfunction

    logic signed [31:0] v0a, v0b, v1a, v1b, v2a, v2b, v3a, v3b;
    logic signed [63:0] acc64;
    logic signed [31:0] acc32;

    always_comb begin
        v0a = decode_fp4_lane(op_a[3:0]);
        v0b = decode_fp4_lane(op_b[3:0]);
        v1a = decode_fp4_lane(op_a[7:4]);
        v1b = decode_fp4_lane(op_b[7:4]);
        v2a = decode_fp4_lane(op_a[11:8]);
        v2b = decode_fp4_lane(op_b[11:8]);
        v3a = decode_fp4_lane(op_a[15:12]);
        v3b = decode_fp4_lane(op_b[15:12]);

        acc64 = 64'sd0;
        acc64 = acc64 + (v0a * v0b) / 64'sd256;
        acc64 = acc64 + (v1a * v1b) / 64'sd256;
        acc64 = acc64 + (v2a * v2b) / 64'sd256;
        acc64 = acc64 + (v3a * v3b) / 64'sd256;
        acc32 = acc64[31:0];
        dp4_res = encode_fp4_val(acc32);
    end

endmodule
