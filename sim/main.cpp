#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vgpu_top.h"

vluint64_t main_time = 0;

double sc_time_stamp() {
    return main_time;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    Vgpu_top* top = new Vgpu_top;
    VerilatedVcdC* tfp = new VerilatedVcdC;
    
    top->trace(tfp, 99);
    tfp->open("sim_trace.vcd");

    top->clk = 0;
    top->rst_n = 0;

    // Simulation loop
    while (!Verilated::gotFinish() && main_time < 200000) {
        // Reset logic
        if (main_time > 10) {
            top->rst_n = 1;
        }

        // Toggle clock
        if ((main_time % 5) == 0) {
            top->clk = !top->clk;
        }

        top->eval();
        tfp->dump(main_time);

        if (top->finished && top->rst_n) {
            printf("Simulation finished at time %lu\n", main_time);
            break;
        }

        main_time++;
    }

    top->final();
    tfp->close();
    delete top;
    return 0;
}

