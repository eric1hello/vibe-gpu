`include "defines.svh"

module regfile (
    input logic clk,
    
    // Read Port (ID Stage)
    input logic [`WARP_ID_WIDTH-1:0] r_warp_id,
    input logic [`REG_ADDR_WIDTH-1:0] raddr1,
    input logic [`REG_ADDR_WIDTH-1:0] raddr2,
    output logic [`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] rdata1,
    output logic [`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] rdata2,
    
    // Write Port (WB Stage)
    input logic [`WARP_ID_WIDTH-1:0] w_warp_id,
    input logic [`THREADS_PER_WARP-1:0] wen,
    input logic [`REG_ADDR_WIDTH-1:0] waddr,
    input logic [`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] wdata
);

    // Storage: [Warp][Thread][Reg]
    // Flattened for simpler Verilog inference usually, but multidim is fine for simulation
    logic [`DATA_WIDTH-1:0] regs [`WARPS_PER_SM-1:0][`THREADS_PER_WARP-1:0][`NUM_REGS-1:0];

    integer t;

    // Write
    always_ff @(posedge clk) begin
        for (t = 0; t < `THREADS_PER_WARP; t = t + 1) begin
            if (wen[t]) begin
                if (waddr != 0) begin 
                    regs[w_warp_id][t][waddr] <= wdata[t];
                end
            end
        end
    end

    // Read (Async/Comb)
    always_comb begin
        for (t = 0; t < `THREADS_PER_WARP; t = t + 1) begin
            rdata1[t] = (raddr1 == 0) ? '0 : regs[r_warp_id][t][raddr1];
            rdata2[t] = (raddr2 == 0) ? '0 : regs[r_warp_id][t][raddr2];
        end
    end

endmodule
