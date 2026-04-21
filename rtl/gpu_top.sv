`include "defines.svh"

module gpu_top (
    input logic clk,
    input logic rst_n,
    output logic finished
);

    // Signals
    logic [`SM_COUNT-1:0] sm_halt_flags;
    
    // Shared Memory (Unified Instruction/Data for simplicity)
    logic [7:0] main_mem [0:(`DMEM_SIZE)-1]; // Byte addressable

    // Interconnect signals
    logic [`SM_COUNT-1:0][`ADDR_WIDTH-1:0] inst_addrs;
    logic [`SM_COUNT-1:0][`DATA_WIDTH-1:0] inst_rdata;

    // SM -> Cache Signals
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0]                 dmem_req; // New
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0]                 dmem_wen;
    /* verilator lint_off UNOPTFLAT */
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0][`ADDR_WIDTH-1:0] dmem_addr;
    /* verilator lint_on UNOPTFLAT */
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] dmem_wdata;
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] dmem_rdata;
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0]                 dmem_miss;

    // Delayed finish logic
    logic all_halted;
    assign all_halted = &sm_halt_flags;
    
    logic [7:0] finish_counter;
    /* verilator lint_off SYNCASYNCNET */
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            finish_counter <= 0;
            finished <= 0;
        end else begin
            if (all_halted) begin
                if (finish_counter < 50) begin
                    finish_counter <= finish_counter + 1;
                end else begin
                    finished <= 1;
                end
            end
        end
    end
    /* verilator lint_on SYNCASYNCNET */

    // Instantiate SMs
    genvar g_i;
    generate
        for (g_i = 0; g_i < `SM_COUNT; g_i = g_i + 1) begin : gen_sm
            sm_core u_sm (
                .clk(clk),
                .rst_n(rst_n),
                .sm_id(g_i[1:0]),
                .inst_addr(inst_addrs[g_i]),
                .inst_data(inst_rdata[g_i]),
                .dmem_req(dmem_req[g_i]),
                .dmem_wen(dmem_wen[g_i]),
                .dmem_addr(dmem_addr[g_i]),
                .dmem_wdata(dmem_wdata[g_i]),
                .dmem_rdata(dmem_rdata[g_i]),
                .dmem_miss(dmem_miss[g_i]),
                .halt_flag(sm_halt_flags[g_i])
            );
        end
    endgenerate

    // ---------------------------------------------------------
    // L1 Cache Integration
    // ---------------------------------------------------------
    // We will instantiate ONE L1 Cache PER THREAD for maximum simplicity and parallelism
    // (simulating a banked cache or private cache).
    // In reality, threads share a cache, but that requires a complex arbiter.
    // For this "vibe" wider SIMD model, let's give each thread its own slice of cache logic
    // but functionally they act as independent requesters to main memory.
    
    // Memory Controller Signals (from Caches)
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0]                 mem_req;
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0]                 mem_wen_l1;
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0][`ADDR_WIDTH-1:0] mem_addr_l1;
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] mem_wdata_l1;
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] mem_rdata_l1;
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0]                 mem_ready;

    // Handshake Logic to ensure 1 req per 1 instruction (handle global stall re-execution issue)
    logic [`SM_COUNT-1:0][`THREADS_PER_WARP-1:0] cache_serviced;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
             for(int i=0; i<`SM_COUNT; i++) cache_serviced[i] <= 0;
        end else begin
             for(int i=0; i<`SM_COUNT; i++) begin
                 for(int j=0; j<`THREADS_PER_WARP; j++) begin
                      if (cache_serviced[i][j]) begin
                           // Clear serviced when req drops (Pipeline moved)
                           if (!dmem_req[i][j]) cache_serviced[i][j] <= 0;
                      end else begin
                           // Set serviced when req active and miss clears (Done)
                           if (dmem_req[i][j] && !dmem_miss[i][j]) begin
                               cache_serviced[i][j] <= 1;
                           end
                      end
                 end
             end
        end
    end

    genvar s, t;
    generate
        for (s = 0; s < `SM_COUNT; s++) begin : gen_cache_sm
            for (t = 0; t < `THREADS_PER_WARP; t++) begin : gen_cache_thread
                l1_cache u_cache (
                    .clk(clk),
                    .rst_n(rst_n),
                    .req(dmem_req[s][t] & !cache_serviced[s][t]), // Gated Request
                    .wen(dmem_wen[s][t]),
                    .addr(dmem_addr[s][t]),
                    .wdata(dmem_wdata[s][t]),
                    .rdata(dmem_rdata[s][t]),
                    /* verilator lint_off PINCONNECTEMPTY */
                    .hit(), // Unused
                    /* verilator lint_on PINCONNECTEMPTY */
                    .miss(dmem_miss[s][t]),
                    
                    .mem_req(mem_req[s][t]),
                    .mem_wen(mem_wen_l1[s][t]),
                    .mem_addr(mem_addr_l1[s][t]),
                    .mem_wdata(mem_wdata_l1[s][t]),
                    .mem_rdata(mem_rdata_l1[s][t]),
                    .mem_ready(mem_ready[s][t])
                );
            end
        end
    endgenerate

    // ---------------------------------------------------------
    // Main Memory Arbiter / Controller
    // ---------------------------------------------------------
    
    logic [3:0] mem_delay_ctr [0:`SM_COUNT-1][0:`THREADS_PER_WARP-1];
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i=0; i<`SM_COUNT; i++)
                for (int j=0; j<`THREADS_PER_WARP; j++) 
                    mem_delay_ctr[i][j] <= 0;
        end else begin
            for (int i=0; i<`SM_COUNT; i++) begin
                for (int j=0; j<`THREADS_PER_WARP; j++) begin
                    mem_ready[i][j] <= 0;
                    
                    if (mem_req[i][j]) begin
                        if (mem_delay_ctr[i][j] == 4) begin // 4 cycle latency
                            mem_ready[i][j] <= 1;
                            mem_delay_ctr[i][j] <= 0;
                            
                            // Perform Memory Access
                            if (mem_wen_l1[i][j]) begin
                                main_mem[mem_addr_l1[i][j] + 0] <= mem_wdata_l1[i][j][7:0];
                                main_mem[mem_addr_l1[i][j] + 1] <= mem_wdata_l1[i][j][15:8];
                                main_mem[mem_addr_l1[i][j] + 2] <= mem_wdata_l1[i][j][23:16];
                                main_mem[mem_addr_l1[i][j] + 3] <= mem_wdata_l1[i][j][31:24];
                            end
                        end else begin
                            mem_delay_ctr[i][j] <= mem_delay_ctr[i][j] + 1;
                        end
                    end else begin
                        mem_delay_ctr[i][j] <= 0;
                    end
                end
            end
        end
    end

    // Async Read for Memory (Data valid when ready asserted)
    always_comb begin
        for (int i=0; i<`SM_COUNT; i++) begin
            for (int j=0; j<`THREADS_PER_WARP; j++) begin
                mem_rdata_l1[i][j] = {
                    main_mem[mem_addr_l1[i][j] + 3],
                    main_mem[mem_addr_l1[i][j] + 2],
                    main_mem[mem_addr_l1[i][j] + 1],
                    main_mem[mem_addr_l1[i][j] + 0]
                };
            end
        end
    end

    // Instruction Fetch (Magic - no cache yet)
    always_comb begin
        for (int i = 0; i < `SM_COUNT; i++) begin
            inst_rdata[i] = {
                main_mem[inst_addrs[i] + 3],
                main_mem[inst_addrs[i] + 2],
                main_mem[inst_addrs[i] + 1],
                main_mem[inst_addrs[i] + 0]
            };
        end
    end

    // Initial Load
    initial begin
        // Initialize memory to 0
        for (int k = 0; k < `DMEM_SIZE; k++) begin
            main_mem[k] = 0;
        end
        // Load program
        $readmemh("../tests/program.hex", main_mem);
    end

    // ---------------------------------------------------------
    // Verification
    // ---------------------------------------------------------
    /* verilator lint_off SYNCASYNCNET */
    always @(posedge finished) begin
            if (finished && rst_n) begin
                int base_c = 2560; // Default for most tests

                // Dump memory to file for python parsing
                int fd;
                fd = $fopen("dump_mem_0_0.txt", "w");
                if (fd != 0) begin
                    // Dump a wider range to cover all tests
                    // Debug/Load/Loop/Div/VecMul use around 2560
                    // Performance uses 2048
                    // MatMul uses 2560
                    // Let's dump 1024 to 4096
                    for (int k=1024; k < 4096; k=k+4) begin
                        int data = {
                           main_mem[k+3], 
                           main_mem[k+2], 
                           main_mem[k+1], 
                           main_mem[k+0]
                        };
                        $fdisplay(fd, "%0d: %0d", k, data);
                    end
                    $fclose(fd);
                end

                $display("\n[GPU] Simulation Finished. Verifying Results...");
            $display("Memory Dump (Offset %d):", base_c);
            
            // Dump 32 rows x 4 cols (128 elements) to cover full grid (16x8)
            for (int r = 0; r < 32; r++) begin
                $write("Row %0d: ", r);
                for (int c = 0; c < 4; c++) begin
                    int addr = base_c + (r * 4 + c) * 4;
                    int data = {
                        main_mem[addr+3], 
                        main_mem[addr+2], 
                        main_mem[addr+1], 
                        main_mem[addr+0]
                    };
                    $write("%4d ", data);
                end
                $write("\n");
            end
        end
    end

endmodule
