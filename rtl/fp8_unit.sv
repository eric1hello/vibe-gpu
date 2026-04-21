module fp8_unit (
    input  logic [7:0] a,
    input  logic [7:0] b,
    output logic [7:0] add_res,
    output logic [7:0] mul_res
);

    // FP8 E4M3 Format: S(1) E(4) M(3) Bias=7
    
    // Unpack Helper
    function automatic logic [8:0] unpack(input logic [7:0] val);
        // Returns {S, E, M_extended} (1+4+4)
        // M_extended includes hidden bit (or 0 if denormal/zero)
        logic s;
        logic [3:0] e;
        logic [2:0] m;
        s = val[7];
        e = val[6:3];
        m = val[2:0];
        if (e == 0) return {s, 4'b0, 1'b0, m}; // Denormal/Zero -> Treat as 0.0 for simplicity (Flush-to-zero)
        else        return {s, e, 1'b1, m};    // Normal -> 1.m
    endfunction

    // ---------------------------------------------------------
    // Multiplier
    // ---------------------------------------------------------
    logic s_a, s_b, s_res;
    logic [3:0] e_a, e_b;
    logic [3:0] m_a, m_b; // 1.mmm
    logic [7:0] m_prod;   // 4x4 = 8 bits
    logic [5:0] e_temp;   // wider to handle overflow
    
    always_comb begin
        {s_a, e_a, m_a} = unpack(a);
        {s_b, e_b, m_b} = unpack(b);
        
        // Sign
        s_res = s_a ^ s_b;
        
        // Zero Check
        if (a == 0 || b == 0 || e_a == 0 || e_b == 0) begin
            mul_res = 8'b0;
        end else begin
            // Exponent
            // E_res = Ea + Eb - Bias
            e_temp = {2'b0, e_a} + {2'b0, e_b} - 6'd7;
            
            // Mantissa Multiply
            m_prod = m_a * m_b; // 1.xxx * 1.xxx = 01.xxxxxx or 1x.xxxxxx (max 1.875*1.875 < 4)
            
            // Normalize
            if (m_prod[7]) begin // 1x.xxxxxx
                e_temp = e_temp + 1;
                m_prod = m_prod >> 1; // Shift to fit 01.xxx
            end
            // Else it is 01.xxxxxx
            
            // Check Overflow/Underflow
            if (e_temp[5] || e_temp[4] || (e_temp > 14)) begin // Overflow
                 if (s_res) mul_res = 8'hFE;
                 else       mul_res = 8'h7E;
            end else if ($signed(e_temp) <= 0) begin // Underflow
                mul_res = 8'b0;
            end else begin
                // Pack
                mul_res = {s_res, e_temp[3:0], m_prod[5:3]};
            end
        end
    end

    // ---------------------------------------------------------
    // Adder
    // ---------------------------------------------------------
    logic s_l, s_s; // Large, Small
    logic [3:0] e_l, e_s;
    logic [3:0] m_l, m_s;
    logic [7:0] m_s_shifted;
    logic [8:0] m_res_uncorr;
    logic [4:0] e_res_add; // 5 bits to hold 16 (overflow)
    
    always_comb begin
        // Unpack again for Adder context
        logic [3:0] ea, eb;
        logic [3:0] ma, mb;
        logic sa, sb;
        {sa, ea, ma} = unpack(a);
        {sb, eb, mb} = unpack(b);
        
        // Handle Zeros
        if (a[6:0] == 0) add_res = b;
        else if (b[6:0] == 0) add_res = a;
        else begin
            // Swap so A is dominant
            if ({ea, ma} >= {eb, mb}) begin
                s_l = sa; e_l = ea; m_l = ma;
                s_s = sb; e_s = eb; m_s = mb;
            end else begin
                s_l = sb; e_l = eb; m_l = mb;
                s_s = sa; e_s = ea; m_s = ma;
            end
            
            // Align Small
            if ((e_l - e_s) > 4) begin
                m_s_shifted = 0; // Shifted out
            end else begin
                m_s_shifted = {m_s, 4'b0} >> (e_l - e_s);
            end
            
            // Add/Sub
            if (s_l == s_s) begin
                m_res_uncorr = {1'b0, m_l, 4'b0} + {1'b0, m_s_shifted};
                
                if (m_res_uncorr[8]) begin // Carry out
                    e_res_add = {1'b0, e_l} + 1;
                    m_res_uncorr = m_res_uncorr >> 1;
                end else begin
                    e_res_add = {1'b0, e_l};
                end
            end else begin
                m_res_uncorr = {1'b0, m_l, 4'b0} - {1'b0, m_s_shifted};
                e_res_add = {1'b0, e_l};
                
                if (m_res_uncorr == 0) begin
                    e_res_add = 0; // Zero result
                end else begin
                    // Normalize
                    if (m_res_uncorr[7]) begin end
                    else if (m_res_uncorr[6]) begin m_res_uncorr <<= 1; e_res_add -= 1; end
                    else if (m_res_uncorr[5]) begin m_res_uncorr <<= 2; e_res_add -= 2; end
                    else if (m_res_uncorr[4]) begin m_res_uncorr <<= 3; e_res_add -= 3; end
                    else if (m_res_uncorr[3]) begin m_res_uncorr <<= 4; e_res_add -= 4; end
                    else e_res_add = 0; // Too small
                end
            end
            
            // Pack
            if (e_res_add == 0 || $signed(e_res_add) <= 0) begin
                add_res = 0;
            end else if (e_res_add >= 15) begin
                // Overflow
                add_res = {s_l, 7'b1111110};
            end else begin
                add_res = {s_l, e_res_add[3:0], m_res_uncorr[6:4]};
            end
        end
    end

endmodule
