`include "defines.svh"

module simt_stack (
    input logic clk,
    input logic rst_n,
    input logic push,
    input logic pop,
    
    input logic push_type, // 0: Divergence (DIV), 1: Convergence (CONV)
    input logic [`THREADS_PER_WARP-1:0] push_mask,
    input logic [`ADDR_WIDTH-1:0] push_pc,
    
    output logic pop_type,
    output logic [`THREADS_PER_WARP-1:0] pop_mask,
    output logic [`ADDR_WIDTH-1:0] pop_pc,
    
    output logic empty,
    output logic full
);

    // Stack entry: {type, mask, pc}
    localparam ENTRY_WIDTH = 1 + `THREADS_PER_WARP + `ADDR_WIDTH;
    
    logic [ENTRY_WIDTH-1:0] stack [`STACK_DEPTH-1:0];
    logic [$clog2(`STACK_DEPTH)-1:0] ptr;

    assign empty = (ptr == 0);
    assign full  = (ptr == 3'd7); // Hardcoded for depth 8

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ptr <= 0;
        end else begin
            if (push && pop && !empty) begin
                // Swap Operation (Overwrite top)
                stack[ptr-1] <= {push_type, push_mask, push_pc};
                // ptr remains unchanged
            end
            else if (push && !full) begin
                stack[ptr] <= {push_type, push_mask, push_pc};
                ptr <= ptr + 1;
            end
            else if (pop && !empty) begin
                ptr <= ptr - 1;
            end
        end
    end

    // Read logic (Peek top)
    assign {pop_type, pop_mask, pop_pc} = (ptr > 0) ? stack[ptr-1] : {1'b0, {`THREADS_PER_WARP{1'b0}}, {`ADDR_WIDTH{1'b0}}};

endmodule
