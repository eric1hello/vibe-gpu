`include "defines.svh"

module l1_cache (
    input logic clk,
    input logic rst_n,

    // Core Interface
    input  logic req,
    input  logic wen,
    input  logic [`ADDR_WIDTH-1:0] addr,
    input  logic [`DATA_WIDTH-1:0] wdata,
    output logic [`DATA_WIDTH-1:0] rdata,
    output logic hit,
    output logic miss,
    
    // Memory Interface
    output logic mem_req,
    output logic mem_wen,
    output logic [`ADDR_WIDTH-1:0] mem_addr,
    output logic [`DATA_WIDTH-1:0] mem_wdata,
    input  logic [`DATA_WIDTH-1:0] mem_rdata,
    input  logic mem_ready
);

    // Direct Mapped Cache
    localparam NUM_LINES = 64;
    localparam INDEX_BITS = 6;
    localparam TAG_BITS = 32 - 2 - INDEX_BITS; // -2 for byte offset (word aligned)
    
    // Cache Arrays
    logic valid [0:NUM_LINES-1];
    logic [TAG_BITS-1:0] tags [0:NUM_LINES-1];
    logic [`DATA_WIDTH-1:0] data [0:NUM_LINES-1];

    // Decomposition
    logic [INDEX_BITS-1:0] index;
    logic [TAG_BITS-1:0] tag;
    
    assign index = addr[INDEX_BITS+1:2]; // Word aligned index
    assign tag   = addr[31:INDEX_BITS+2];

    // Hit Detection
    logic tag_match;
    assign tag_match = (tags[index] === tag);
    assign hit = valid[index] && tag_match && req;

    // State Machine for Miss Handling & Write Buffer
    typedef enum logic [1:0] {
        IDLE,
        WAIT_MEM,
        UPDATE, // For Read Refill
        DONE    // For Write Completion
    } state_t;
    state_t state, next_state;

    // Stall Logic (Output 'miss' causes SM to stall)
    // Stall on:
    // 1. Read Miss (req && !hit) until Refill (UPDATE state)
    // 2. Write (req && wen) until Memory Done (DONE state)
    // Note: For Read, logic simplifies to (!hit) because hit becomes 1 in UPDATE.
    // For Write, hit/miss doesn't matter, we always go to memory (Write-Through).
    assign miss = (req && !wen && !hit) || (req && wen && state != DONE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            for (int i=0; i<NUM_LINES; i++) valid[i] <= 0;
        end else begin
            state <= next_state;
            
            // Cache Update Logic
            // Update on Write (when sending to memory)
            // Can update immediately in IDLE or wait?
            // Let's update immediately to keep it simple, but stall until memory done.
            if (state == IDLE && req && wen) begin
                valid[index] <= 1;
                tags[index] <= tag;
                data[index] <= wdata;
            end
            else if (state == UPDATE) begin // Read Refill
                valid[index] <= 1;
                tags[index] <= tag;
                data[index] <= mem_rdata;
            end
        end
    end

    // Read Data
    assign rdata = data[index];

    // Next State & Output Logic
    always_comb begin
        next_state = state;
        mem_req = 0;
        mem_wen = 0;
        mem_addr = addr;
        mem_wdata = wdata;

        case (state)
            IDLE: begin
                if (req) begin
                    if (wen) begin
                        // Write (Hit or Miss): Go to Memory
                        next_state = WAIT_MEM;
                        mem_req = 1;
                        mem_wen = 1;
                    end else if (!hit) begin
                        // Read Miss: Go to Memory
                        next_state = WAIT_MEM;
                        mem_req = 1;
                    end
                end
            end
            
            WAIT_MEM: begin
                mem_req = 1;
                if (wen) mem_wen = 1;
                
                if (mem_ready) begin
                    if (wen) 
                        next_state = DONE;
                    else
                        next_state = UPDATE;
                end
            end
            
            UPDATE: begin
                // Cache Array updated in always_ff
                // Hit becomes 1. Miss becomes 0 (for read).
                next_state = IDLE;
            end
            
            DONE: begin
                // Write completed. Miss becomes 0 (for write).
                next_state = IDLE;
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end

endmodule
