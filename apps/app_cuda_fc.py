import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tools.vibe_cuda as cuda

# Fully Connected Layer Configuration
# Input: 16 features
# Output: 16 neurons
# Batch: 1
# W: 16x16
# B: 16
# Total Threads needed: 16 (one per output neuron)

N_IN = 16
N_OUT = 16

@cuda.jit(block_dim=32)
def fc_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr):
    # Global ID corresponds to the output neuron index
    gid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    gid = gid * bdim
    tid = cuda.threadIdx.x
    gid = gid + tid
    
    # Check bounds (we launch 32 threads but only need 16)
    limit = 16
    
    # If gid < 16
    # Vibe ASM doesn't support complex "if gid < limit" block structure perfectly inside other flows yet 
    # (nested stack operations might be tricky if we return early).
    # Instead, we mask operations or just compute zeros for out-of-bound threads.
    # Or simpler: Just launch 1 block of 32, use first 16.
    
    # We'll just iterate and compute. If gid >= 16, we are writing to garbage or we handle it.
    # Let's implement specific check.
    
    acc = 0
    
    # Dot Product: Loop over input features
    for k in range(16): # N_IN
        # Load Input[k]
        val_in = input_ptr[k]
        
        # Load Weight[gid * 16 + k]
        # w_offset = gid * 16 + k
        w_idx = gid << 4 # * 16
        w_idx = w_idx + k
        
        val_w = weight_ptr[w_idx]
        
        # MAC
        prod = val_in * val_w
        acc = acc + prod
        
    # Add Bias
    val_b = bias_ptr[gid]
    acc = acc + val_b
    
    # ReLU Activation: y = max(0, acc)
    # Check if acc < 0. 
    # We don't have signed comparison operator exposed clearly in AST yet? 
    # The ALU does SUB. If acc < 0, MSB is 1.
    # But let's assume for this test inputs are such that we can test simple logic.
    # Or add a constant to check "if acc > threshold".
    # Let's rely on assembler BEQ/BNE logic or implement `if acc < 0` via high-level `if`.
    
    # Implementation of ReLU using if:
    # if acc < 0: acc = 0
    # Currently `vibe_cuda.py` maps `if` to `BEQ/BNE`.
    # It doesn't support `BLT` (Branch Less Than).
    # Workaround: ALU shift to check sign bit?
    # sign = acc >> 31. If sign == 1, it's negative.
    
    sign = acc >> 31
    is_neg = sign & 1
    if is_neg:
        acc = 0
        
    # Store Output
    # Only store if gid < 16? 
    # We can just store for all 32 threads, we have memory space.
    output_ptr[gid] = acc

def main():
    # Memory Layout
    # Code: 0 - ~1024
    # Data Start: 1024
    
    base_in = 1024                  # Size: 16 * 4 = 64 bytes
    base_w  = base_in + 64          # Size: 16*16*4 = 1024 bytes
    base_b  = base_w + 1024         # Size: 16 * 4 = 64 bytes
    base_out = 2560                 # Updated to match gpu_top default dump location
    
    print(f"Compiling FC Kernel... In@{base_in}, W@{base_w}, B@{base_b}, Out@{base_out}")
    
    asm = fc_kernel(base_in, base_w, base_b, base_out)
    
    # Data Generation
    # Input: [1, 1, 1, ...]
    vals_in = [1] * N_IN
    
    # Weights: Identity-like or simple pattern
    # Let's make row 0: all 1s -> dot = 16
    # Row 1: all 2s -> dot = 32
    # ...
    # Also mix some negative weights to test ReLU
    vals_w = []
    for r in range(N_OUT):
        for c in range(N_IN):
            if r < 8:
                vals_w.append(1) # Rows 0-7: Sum = 16
            else:
                vals_w.append(-2) # Rows 8-15: Sum = -32
                
    # Bias
    # Bias = 0 for positive tests
    # Bias = 10 for negative tests (to see if it stays negative or becomes positive)
    vals_b = []
    for r in range(N_OUT):
        if r < 8:
            vals_b.append(0) 
        else:
            vals_b.append(10) # -32 + 10 = -22. ReLU -> 0.
            
    # Expected Output
    # Rows 0-7: (1*1)*16 + 0 = 16. ReLU(16) = 16.
    # Rows 8-15: (1*-2)*16 + 10 = -32 + 10 = -22. ReLU(-22) = 0.
    
    vals_out_gold = []
    for r in range(N_OUT):
        if r < 8:
            vals_out_gold.append(16)
        else:
            vals_out_gold.append(0)
            
    # Inject Data
    asm.base_data_addr = base_in
    asm.data_offset = 0
    
    asm.data(vals_in)
    asm.data(vals_w)
    asm.data(vals_b)
    # Padding for output
    asm.data([0] * 32) 
    
    asm.write_hex("tests/program.hex")
    
    with open("tests/gold.txt", "w") as f:
        for r in range(8): # Dump 8 rows of memory map
            # Our output is at base_out. 
            # Verification logic in gpu_top usually dumps from a fixed offset `base_c`.
            # We need to ensure gpu_top dumps `base_out`.
            # Let's generate gold for the specific memory region we care about.
            pass
            
    # We will update test_suite to check the specific output array
    print("Done.")

if __name__ == "__main__":
    main()

