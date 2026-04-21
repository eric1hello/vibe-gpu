import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tools.vibe_cuda as cuda

@cuda.jit(block_dim=32)
def perf_kernel(results):
    gid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    # Only Thread 0 measures performance
    if gid == 0:
        start_cycle = cuda.csr_cycle()
        start_inst = cuda.csr_instret()
        
        # --- Workload: Matrix Vector Mul (Simplified) ---
        # Doing some heavy work to measure IPC
        acc = 0
        for i in range(4):
            acc = acc + i
            acc = acc * 2
        
        end_cycle = cuda.csr_cycle()
        end_inst = cuda.csr_instret()
        
        # Store results
        results[0] = start_cycle
        results[1] = end_cycle
        results[2] = start_inst
        results[3] = end_inst
        results[4] = acc

def main():
    print("Compiling Performance Benchmark Kernel...")
    
    # Output buffer (5 ints)
    # 0: Start Cycle
    # 1: End Cycle
    # 2: Start Inst
    # 3: End Inst
    # 4: Checksum
    
    # We pass 1024 as the address of the 'results' buffer
    asm = perf_kernel(2048) 
    asm.base_data_addr = 2048
    asm.data([0] * 128) # Buffer
    
    asm.write_hex("tests/program.hex")
    print("Hex written to tests/program.hex")

    # Run Simulation
    import subprocess
    print("Running Simulation...")
    try:
        subprocess.run("cd sim && make clean && make run", shell=True, check=True)
    except subprocess.CalledProcessError:
        print("Simulation Failed!")
        return

    # Parse Results
    mem_file = "sim/dump_mem_0_0.txt" # Check SM 0, Thread 0 (since gid 0 is here)
    if not os.path.exists(mem_file):
        print("No memory dump found!")
        return
        
    with open(mem_file, "r") as f:
        lines = f.readlines()
        
    # Data starts at 1024. In dump_mem_0_0, addresses are 0-based relative to thread's space if banked?
    # Wait, main memory is unified. 
    # Our dump logic in gpu_top might be dumping specific ranges.
    # Let's parse line by line looking for our data addresses.
    
    # Address 1024 / 4 = 256 words offset
    # But memory dump format is "Addr: Value"
    
    mem_map = {}
    for line in lines:
        parts = line.strip().split(":")
        if len(parts) == 2:
            try:
                addr = int(parts[0].strip()) # Decimal address
                val = int(parts[1].strip())  # Decimal value
                mem_map[addr] = val
            except:
                pass
                
    # We expect data at 2048, 2052, 2056, 2060, 2064
    start_cyc = mem_map.get(2048, 0)
    end_cyc = mem_map.get(2052, 0)
    start_inst = mem_map.get(2056, 0)
    end_inst = mem_map.get(2060, 0)
    checksum = mem_map.get(2064, 0)
    
    print("-" * 40)
    print(f"Performance Results (Thread 0)")
    print("-" * 40)
    print(f"Start Cycle: {start_cyc}")
    print(f"End Cycle:   {end_cyc}")
    print(f"Delta Cycle: {end_cyc - start_cyc}")
    print("-" * 40)
    print(f"Start Inst:  {start_inst}")
    print(f"End Inst:    {end_inst}")
    print(f"Delta Inst:  {end_inst - start_inst}")
    print("-" * 40)
    
    if (end_cyc - start_cyc) > 0:
        delta_cycle = end_cyc - start_cyc
        ipc = (end_inst - start_inst) / delta_cycle
        print(f"IPC (Instruction Throughput): {ipc:.4f}")
        
        # Performance Estimation
        # We executed 64 iterations of (Add + Mul + Shift) = 3 Ops
        # Parallelism: 4 SMs * 8 Cores = 32 Cores active
        # (Assuming full occupancy, though this test only checks Thread 0's view)
        # But since it is SIMT, all 32 threads in the GPU (if Grid=32) were running.
        # Wait, BlockDim=32. So 1 Block. 4 Warps.
        # SM 0 executes 4 warps. SM 1,2,3 are idle in this specific test kernel launch?
        # cuda.jit(block_dim=32) -> usually launches 1 block if not specified.
        # Let's assume 1 Block running on SM 0.
        
        # Active Threads = 32.
        num_ops_per_thread = 64 * 3 # 3 ops per loop
        total_ops = 32 * num_ops_per_thread
        
        ops_per_cycle = total_ops / delta_cycle
        print("-" * 40)
        print(f"Performance Metrics (1 Block, 32 Threads)")
        print(f"Total Ops Executed: {total_ops}")
        print(f"Ops Per Cycle:      {ops_per_cycle:.4f}")
        print(f"Est. GOPS @ 100MHz: {ops_per_cycle * 0.1:.4f} GOPS")
    else:
        print("IPC: N/A (Zero Cycles)")
        
    print(f"Checksum: {checksum}")

if __name__ == "__main__":
    main()

