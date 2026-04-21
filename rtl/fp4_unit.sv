// ============================================================================
// FP4 运算单元（演进版 Vibe-GPU）
// 格式：E2M1（OCP / 微缩放量化常用子集）
//   - 位域：S(1) | E(2) | M(1)，指数偏置 bias = 1
//   - E = 0：按零处理（非规格数 flush，简化硬件）
//   - 正规数：值 = (-1)^S * 2^(E-1) * (1 + M/2)，E ∈ {1,2,3}
// 内部使用定点（FIX=256）完成乘加后再 pack，便于与软件黄金值逐位对齐。
// ============================================================================

module fp4_unit (
    input  logic [3:0] a,
    input  logic [3:0] b,
    output logic [3:0] add_res,
    output logic [3:0] mul_res
);

    localparam int unsigned FIX = 256;

    // 将有符号定点值编码为 4 位 FP4（与 tools/fp4_soft.py 一致）
    function automatic logic [3:0] encode_mag(input logic sign_in, input logic signed [31:0] mag_fix);
        logic signed [31:0] absv;
        logic [1:0] best_e;
        logic best_m;
        integer best_err, err, t;
        begin
            if (mag_fix == 0) return 4'b0000;
            absv = mag_fix < 0 ? -mag_fix : mag_fix;
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
            encode_mag = {sign_in, best_e[1:0], best_m};
        end
    endfunction

    // 解码为无符号幅度（定点）；参数为 FP4 的 E、M 位域 [2:0]（不含符号位）
    function automatic logic signed [31:0] decode_mag_bits(input logic [2:0] f);
        logic [1:0] e;
        logic m;
        begin
            e = f[2:1];
            m = f[0];
            if (e == 2'b00)
                decode_mag_bits = 0;
            else
                decode_mag_bits = ($signed(32'(32'd1 << (e - 1))) * $signed(32'(32'd2 + {31'b0, m})) * FIX) >>> 1;
        end
    endfunction

    logic signed [31:0] ma_a, mb_b, aa, ab;
    logic signed [63:0] prod_wide;
    logic signed [31:0] prod_fix;
    logic sa, sb;

    always_comb begin
        // ---------- 乘法：幅度解码后恢复符号，再定点相乘 ----------
        sa = a[3];
        sb = b[3];
        ma_a = decode_mag_bits(a[2:0]);
        mb_b = decode_mag_bits(b[2:0]);
        if (a[2:1] == 2'b00 || b[2:1] == 2'b00) begin
            mul_res = 4'b0000;
        end else begin
            ma_a = sa ? -ma_a : ma_a;
            mb_b = sb ? -mb_b : mb_b;
            prod_wide = ma_a * mb_b;
            // 有符号除法，商为 32 位，与 Python/Verilog 向零截断一致
            prod_fix = 32'($signed(prod_wide) / 64'sd256);
            mul_res = encode_mag(prod_fix < 0, prod_fix);
        end

        // ---------- 加法 ----------
        sa = a[3];
        sb = b[3];
        ma_a = decode_mag_bits(a[2:0]);
        mb_b = decode_mag_bits(b[2:0]);
        if (a[2:1] == 2'b00) begin
            add_res = b;
        end else if (b[2:1] == 2'b00) begin
            add_res = a;
        end else begin
            aa = sa ? -ma_a : ma_a;
            ab = sb ? -mb_b : mb_b;
            add_res = encode_mag((aa + ab) < 0, aa + ab);
        end
    end

endmodule
