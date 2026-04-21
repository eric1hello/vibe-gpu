`include "defines.svh"

module sm_core (
    input logic clk,
    input logic rst_n,
    input logic [1:0] sm_id,

    // Instruction Memory Interface
    output logic [`ADDR_WIDTH-1:0] inst_addr,
    input  logic [`DATA_WIDTH-1:0] inst_data,

    // Data Memory Interface
    output logic [`THREADS_PER_WARP-1:0]                 dmem_req, // New: Request signal (Read or Write)
    output logic [`THREADS_PER_WARP-1:0]                 dmem_wen,
    output logic [`THREADS_PER_WARP-1:0][`ADDR_WIDTH-1:0] dmem_addr,
    output logic [`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] dmem_wdata,
    input  logic [`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] dmem_rdata,
    input  logic [`THREADS_PER_WARP-1:0] dmem_miss,

    output logic halt_flag // Global halt (all warps halted)
);

    // =========================================================================
    // Architectural State (Per Warp)
    // =========================================================================
    logic [`ADDR_WIDTH-1:0] pc_table [`WARPS_PER_SM-1:0];
    logic [`THREADS_PER_WARP-1:0] exec_mask_table [`WARPS_PER_SM-1:0];
    logic [`WARPS_PER_SM-1:0] warp_active; // Bitmask of active warps (not halted)
    logic [`WARPS_PER_SM-1:0] warp_halted; // Track halted status
    
    // Scheduler State
    logic [`WARP_ID_WIDTH-1:0] current_fetch_warp;
    logic [`WARP_ID_WIDTH-1:0] next_fetch_warp;

    // Performance Counters
    logic [31:0] counter_cycle;
    logic [31:0] counter_instret;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter_cycle <= 0;
            counter_instret <= 0;
        end else begin
            counter_cycle <= counter_cycle + 1;
            // Instret increment: valid inst in MEM/WB, not stall, not bubble
            if (mem_wb_valid && !stall_all && mem_wb_opcode != `OP_NOP && mem_wb_opcode != `OP_HALT) 
                counter_instret <= counter_instret + 1;
        end
    end

    // =========================================================================
    // Pipeline Registers
    // =========================================================================
    
    // IF/ID
    logic if_id_valid;
    logic [`WARP_ID_WIDTH-1:0] if_id_warp;
    logic [`ADDR_WIDTH-1:0] if_id_pc;
    logic [`DATA_WIDTH-1:0] if_id_inst;
    logic [`THREADS_PER_WARP-1:0] if_id_mask;

    // ID/EX
    logic id_ex_valid;
    logic [`WARP_ID_WIDTH-1:0] id_ex_warp;
    logic [`ADDR_WIDTH-1:0] id_ex_pc;
    logic [5:0] id_ex_opcode;
    logic [`REG_ADDR_WIDTH-1:0] id_ex_rd;
    /* verilator lint_off UNUSED */
    logic [`REG_ADDR_WIDTH-1:0] id_ex_rs1, id_ex_rs2;
    /* verilator lint_on UNUSED */
    logic [`DATA_WIDTH-1:0] id_ex_imm;
    logic [`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] id_ex_rdata1, id_ex_rdata2;
    logic [`THREADS_PER_WARP-1:0] id_ex_mask;

    // EX/MEM
    logic ex_mem_valid;
    logic [`WARP_ID_WIDTH-1:0] ex_mem_warp;
    /* verilator lint_off UNUSED */
    logic [`ADDR_WIDTH-1:0] ex_mem_pc;
    /* verilator lint_on UNUSED */
    logic [5:0] ex_mem_opcode;
    logic [`REG_ADDR_WIDTH-1:0] ex_mem_rd;
    logic [`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] ex_mem_alu_res;
    logic [`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] ex_mem_wdata;
    logic [`THREADS_PER_WARP-1:0] ex_mem_mask;
    logic ex_mem_reg_write;

    // MEM/WB
    logic mem_wb_valid;
    logic [`WARP_ID_WIDTH-1:0] mem_wb_warp;
    logic [5:0] mem_wb_opcode;
    logic [`REG_ADDR_WIDTH-1:0] mem_wb_rd;
    logic [`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] mem_wb_res;
    logic [`THREADS_PER_WARP-1:0] mem_wb_mask;
    logic mem_wb_reg_write;

    // Control Signals
    logic stall_all;
    logic flush_branch;
    /* verilator lint_off UNUSED */
    logic stall_hazard; // Global hazard stall signal
    /* verilator lint_on UNUSED */
    logic hazard_raw;
    logic hazard_struct;

    assign stall_all = (|(dmem_miss & ex_mem_mask) && ex_mem_valid && (ex_mem_opcode == `OP_LDW || ex_mem_opcode == `OP_STW));

    // =========================================================================
    // SIMT Stack Instances
    // =========================================================================
    
    logic stack_push [`WARPS_PER_SM-1:0];
    logic stack_pop [`WARPS_PER_SM-1:0];
    logic stack_push_type [`WARPS_PER_SM-1:0];
    logic [`THREADS_PER_WARP-1:0] stack_push_mask [`WARPS_PER_SM-1:0];
    logic [`ADDR_WIDTH-1:0] stack_push_pc [`WARPS_PER_SM-1:0];
    
    logic stack_pop_type [`WARPS_PER_SM-1:0];
    logic [`THREADS_PER_WARP-1:0] stack_pop_mask [`WARPS_PER_SM-1:0];
    logic [`ADDR_WIDTH-1:0] stack_pop_pc [`WARPS_PER_SM-1:0];
    logic stack_empty [`WARPS_PER_SM-1:0];
    /* verilator lint_off UNUSED */
    logic stack_full [`WARPS_PER_SM-1:0];
    /* verilator lint_on UNUSED */

    genvar w;
    generate
        for (w = 0; w < `WARPS_PER_SM; w = w + 1) begin : gen_stacks
            simt_stack u_simt_stack (
                .clk(clk),
                .rst_n(rst_n),
                .push(stack_push[w] & !stall_all),
                .pop(stack_pop[w] & !stall_all),
                .push_type(stack_push_type[w]),
                .push_mask(stack_push_mask[w]),
                .push_pc(stack_push_pc[w]),
                .pop_type(stack_pop_type[w]),
                .pop_mask(stack_pop_mask[w]),
                .pop_pc(stack_pop_pc[w]),
                .empty(stack_empty[w]),
                .full(stack_full[w])
            );
        end
    endgenerate

    // Scoreboard
    logic [`NUM_REGS-1:0] scoreboard [`WARPS_PER_SM-1:0];

    // =========================================================================
    // IF Stage: Scheduler
    // =========================================================================

    always_comb begin
        next_fetch_warp = current_fetch_warp;
        for (int k = 1; k <= `WARPS_PER_SM; k++) begin
            logic [`WARP_ID_WIDTH-1:0] candidate;
            candidate = (current_fetch_warp + k[`WARP_ID_WIDTH-1:0]); 
            if (warp_active[candidate] && !warp_halted[candidate]) begin
                next_fetch_warp = candidate;
                break;
            end
        end
    end

    assign inst_addr = pc_table[current_fetch_warp];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int k=0; k<`WARPS_PER_SM; k++) begin
                pc_table[k] <= 0;
                exec_mask_table[k] <= {`THREADS_PER_WARP{1'b1}};
                warp_halted[k] <= 0;
            end
            current_fetch_warp <= 0;
            warp_active <= {`WARPS_PER_SM{1'b1}};
        end else if (!stall_all) begin
            // If hazard_struct (bubble insertion), we must STALL the front-end
            // If hazard_raw, we must STALL the front-end
            if (!hazard_struct && !hazard_raw) begin
                current_fetch_warp <= next_fetch_warp;
                
                if (!flush_branch || (id_ex_warp != current_fetch_warp))
                    pc_table[current_fetch_warp] <= pc_table[current_fetch_warp] + 4;
            end

            // Branch flush overrides
            if (flush_branch) begin
                 pc_table[id_ex_warp] <= next_pc_ex;
                 exec_mask_table[id_ex_warp] <= next_exec_mask_ex;
            end
        end
    end

    // IF/ID Pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            if_id_valid <= 0;
            if_id_warp <= 0;
            if_id_pc <= 0;
            if_id_inst <= 0;
            if_id_mask <= 0;
        end else if (!stall_all) begin
            if (hazard_struct || hazard_raw) begin
                // Stall IF/ID: Keep current instruction
            end else if (flush_branch && id_ex_warp == current_fetch_warp) begin
                if_id_valid <= 0;
            end else if (warp_halted[current_fetch_warp]) begin
                if_id_valid <= 0;
            end else begin
                if_id_valid <= 1;
                if_id_warp <= current_fetch_warp;
                if_id_pc <= pc_table[current_fetch_warp];
                if_id_inst <= inst_data;
                if_id_mask <= exec_mask_table[current_fetch_warp];
            end
        end
    end

    // =========================================================================
    // ID Stage
    // =========================================================================

    logic [5:0] id_opcode;
    logic [`REG_ADDR_WIDTH-1:0] id_rd, id_rs1, id_rs2;
    logic [`DATA_WIDTH-1:0] id_imm;
    
    // New Instruction Encoding Decoding
    assign id_opcode = if_id_inst[31:26];
    assign id_rd     = if_id_inst[25:21]; // 5 bits
    assign id_rs1    = if_id_inst[20:16]; // 5 bits
    assign id_rs2    = if_id_inst[15:11]; // 5 bits
    assign id_imm    = {{21{if_id_inst[10]}}, if_id_inst[10:0]}; // 11 bits Imm

    // Regfile Read
    logic [`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] rf_rdata1, rf_rdata2;
    regfile u_regfile (
        .clk(clk),
        .r_warp_id(if_id_warp),
        .raddr1(id_rs1),
        .raddr2(id_rs2),
        .rdata1(rf_rdata1),
        .rdata2(rf_rdata2),
        
        .w_warp_id(mem_wb_warp),
        .wen(wb_wen),
        .waddr(mem_wb_rd),
        .wdata(mem_wb_res)
    );

    // Hazard Detection
    always_comb begin
        hazard_raw = 0;
        if (if_id_valid) begin
            if (id_rs1 != 0 && scoreboard[if_id_warp][id_rs1]) hazard_raw = 1;
             if ((id_opcode == `OP_ADD || id_opcode == `OP_SUB || id_opcode == `OP_MUL || id_opcode == `OP_AND || id_opcode == `OP_OR || id_opcode == `OP_STW || id_opcode == `OP_BEQ || id_opcode == `OP_BNE || id_opcode == `OP_FADD || id_opcode == `OP_FMUL || id_opcode == `OP_FADD4 || id_opcode == `OP_FMUL4 || id_opcode == `OP_TCDP4 || id_opcode == `OP_SLL || id_opcode == `OP_SRL) && id_rs2 != 0 && scoreboard[if_id_warp][id_rs2]) hazard_raw = 1;
        end
        // Structural Hazard: MEM op in EX/MEM
        hazard_struct = 0;
        if (ex_mem_valid && (ex_mem_opcode == `OP_LDW || ex_mem_opcode == `OP_STW)) hazard_struct = 1;
        
        stall_hazard = hazard_raw | hazard_struct;
    end

    // ID/EX Pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            id_ex_valid <= 0;
            id_ex_opcode <= `OP_NOP;
            id_ex_warp <= 0;
            // ...
        end else if (!stall_all) begin
            if (hazard_struct) begin
                // Stall ID/EX: Keep current instruction
                // Do NOT clear valid
            end else if (hazard_raw || (flush_branch && id_ex_warp == if_id_warp)) begin
                // Insert Bubble (NOP)
                id_ex_valid <= 0;
                id_ex_opcode <= `OP_NOP;
            end else begin
                // Normal Advance
                id_ex_valid <= if_id_valid;
                id_ex_warp <= if_id_warp;
                id_ex_pc <= if_id_pc;
                id_ex_opcode <= id_opcode;
                id_ex_rd <= id_rd;
                id_ex_rs1 <= id_rs1;
                id_ex_rs2 <= id_rs2;
                id_ex_imm <= id_imm;
                id_ex_rdata1 <= rf_rdata1;
                id_ex_rdata2 <= rf_rdata2;
                id_ex_mask <= if_id_mask;
                
                // Scoreboard Set (Only if advancing)
                if (if_id_valid && !hazard_raw && id_rd != 0 && (id_opcode != `OP_STW && id_opcode != `OP_BEQ && id_opcode != `OP_BNE && id_opcode != `OP_JOIN && id_opcode != `OP_HALT)) begin
                    scoreboard[if_id_warp][id_rd] <= 1;
                end
            end
        end
    end

    // =========================================================================
    // EX Stage
    // =========================================================================
    
    logic [`ADDR_WIDTH-1:0] next_pc_ex;
    logic [`THREADS_PER_WARP-1:0] next_exec_mask_ex;

    logic [`THREADS_PER_WARP-1:0][`DATA_WIDTH-1:0] alu_res;
    genvar i;
    generate
        for (i = 0; i < `THREADS_PER_WARP; i = i + 1) begin : gen_alu
            alu u_alu (
                .opcode(id_ex_opcode),
                .op_a(id_ex_rdata1[i]),
                .op_b(id_ex_rdata2[i]),
                .imm(id_ex_imm),
                .thread_id(i[2:0]),
                .warp_id(id_ex_warp),
                .sm_id(sm_id),
                .csr_cycle(counter_cycle),
                .csr_instret(counter_instret),
                .result(alu_res[i])
            );
        end
    endgenerate

    // Branch Logic
    always_comb begin
        flush_branch = 0;
        next_pc_ex = id_ex_pc; 
        next_exec_mask_ex = id_ex_mask;
        
        for(int k=0; k<`WARPS_PER_SM; k++) begin
            stack_push[k] = 0; stack_pop[k] = 0;
            stack_push_type[k] = 0; stack_push_mask[k] = 0; stack_push_pc[k] = 0;
        end

        if (id_ex_valid && !stall_hazard) begin // Fix: Gate branch logic with stall_hazard to prevent re-execution
            logic [`THREADS_PER_WARP-1:0] t_cond, t_taken, t_not_taken;
            for (int t=0; t<`THREADS_PER_WARP; t++) begin
                case(id_ex_opcode)
                   `OP_BEQ: t_cond[t] = (id_ex_rdata1[t] == id_ex_rdata2[t]);
                   `OP_BNE: t_cond[t] = (id_ex_rdata1[t] != id_ex_rdata2[t]);
                   default: t_cond[t] = 0;
                endcase
            end
            t_taken = id_ex_mask & t_cond;
            t_not_taken = id_ex_mask & ~t_cond;

            if (id_ex_opcode == `OP_BEQ || id_ex_opcode == `OP_BNE) begin
                if (|t_taken && |t_not_taken) begin 
                    flush_branch = 1;
                    stack_push[id_ex_warp] = 1;
                    stack_push_type[id_ex_warp] = 0; // DIV
                    stack_push_mask[id_ex_warp] = t_not_taken;
                    stack_push_pc[id_ex_warp]   = id_ex_pc + 4;
                    next_exec_mask_ex = t_taken;
                    next_pc_ex = id_ex_pc + ({{22{id_ex_imm[9]}}, id_ex_imm[9:0]} * 4); 
                    next_pc_ex = id_ex_pc + (id_ex_imm * 4);
                end else if (|t_taken) begin 
                    flush_branch = 1;
                    next_pc_ex = id_ex_pc + (id_ex_imm * 4);
                    next_exec_mask_ex = t_taken;
                end
            end 
            else if (id_ex_opcode == `OP_JOIN) begin
                flush_branch = 1;
                if (!stack_empty[id_ex_warp]) begin
                    if (stack_pop_type[id_ex_warp] == 0) begin 
                        stack_pop[id_ex_warp] = 1;
                        stack_push[id_ex_warp] = 1;
                        stack_push_type[id_ex_warp] = 1; 
                        stack_push_mask[id_ex_warp] = id_ex_mask;
                        stack_push_pc[id_ex_warp] = id_ex_pc + 4;
                        next_exec_mask_ex = stack_pop_mask[id_ex_warp];
                        next_pc_ex = stack_pop_pc[id_ex_warp];
                    end else if (stack_pop_pc[id_ex_warp] == id_ex_pc + 4) begin 
                        stack_pop[id_ex_warp] = 1;
                        next_exec_mask_ex = id_ex_mask | stack_pop_mask[id_ex_warp];
                        next_pc_ex = id_ex_pc + 4;
                    end else begin
                        // Mismatch PC (Nested Scope): Do not merge, treat as NOP
                        next_pc_ex = id_ex_pc + 4;
                        next_exec_mask_ex = id_ex_mask;
                    end
                end else begin
                    next_pc_ex = id_ex_pc + 4;
                    next_exec_mask_ex = id_ex_mask;
                end
            end
        end
    end

    // EX/MEM Pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ex_mem_valid <= 0;
            ex_mem_opcode <= `OP_NOP;
            // ...
        end else if (!stall_all) begin
            if (hazard_struct) begin
                // Insert Bubble into EX/MEM
                ex_mem_valid <= 0;
                ex_mem_opcode <= `OP_NOP;
            end else begin
                ex_mem_valid <= id_ex_valid;
                ex_mem_warp <= id_ex_warp;
                ex_mem_pc <= id_ex_pc;
                ex_mem_opcode <= id_ex_opcode;
                ex_mem_rd <= id_ex_rd;
                ex_mem_alu_res <= alu_res;
                ex_mem_wdata <= id_ex_rdata2;
                ex_mem_mask <= id_ex_mask;
                ex_mem_reg_write <= (id_ex_opcode != `OP_STW && id_ex_opcode != `OP_BEQ && id_ex_opcode != `OP_BNE && id_ex_opcode != `OP_JOIN && id_ex_opcode != `OP_NOP && id_ex_opcode != `OP_HALT);
            end
        end
    end

    // =========================================================================
    // MEM Stage
    // =========================================================================
    
    always_comb begin
        dmem_req = 0;
        dmem_wen = 0;
        dmem_addr = 0;
        dmem_wdata = 0;
        if (ex_mem_valid) begin
             if (ex_mem_opcode == `OP_STW) begin
                 dmem_req = ex_mem_mask;
                 dmem_wen = ex_mem_mask;
                 dmem_addr = ex_mem_alu_res;
                 dmem_wdata = ex_mem_wdata;
             end else if (ex_mem_opcode == `OP_LDW) begin
                 dmem_req = ex_mem_mask;
                 dmem_addr = ex_mem_alu_res;
             end
        end
    end

    // MEM/WB Pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_wb_valid <= 0;
            mem_wb_opcode <= `OP_NOP;
        end else if (!stall_all) begin
            mem_wb_valid <= ex_mem_valid;
            mem_wb_warp <= ex_mem_warp;
            mem_wb_opcode <= ex_mem_opcode;
            mem_wb_rd <= ex_mem_rd;
            mem_wb_mask <= ex_mem_mask;
            if (ex_mem_opcode == `OP_LDW) mem_wb_res <= dmem_rdata;
            else mem_wb_res <= ex_mem_alu_res;
            mem_wb_reg_write <= ex_mem_reg_write;
        end
    end

    // =========================================================================
    // WB Stage
    // =========================================================================
    
    logic [`THREADS_PER_WARP-1:0] wb_wen;
    always_comb begin
        wb_wen = 0;
        if (mem_wb_valid && mem_wb_reg_write) wb_wen = mem_wb_mask;
    end

    // Scoreboard Clear & Halt
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset handled at top
        end else if (!stall_all) begin
            if (mem_wb_valid && mem_wb_reg_write && mem_wb_rd != 0) begin
                scoreboard[mem_wb_warp][mem_wb_rd] <= 0;
            end

            if (mem_wb_valid && mem_wb_opcode == `OP_HALT) begin
                warp_halted[mem_wb_warp] <= 1;
            end
        end
    end
    
    assign halt_flag = &warp_halted;

endmodule
