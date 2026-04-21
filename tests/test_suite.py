import os
import subprocess
import sys
import re

def _py(script_path: str) -> str:
    """使用当前解释器运行 apps 脚本，避免 Windows 上 python3 指向商店占位符。"""
    return f'"{sys.executable}" {script_path}'


# Helper to run command
def run_cmd(cmd):
    print(f"[EXEC] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAIL] Command failed with return code {result.returncode}")
        print(result.stderr)
        return False, result.stdout
    return True, result.stdout

def parse_simulation_output(output):
    # Prefer loading from dump file if available
    if os.path.exists("sim/dump_mem_0_0.txt"):
        mem_map = {}
        with open("sim/dump_mem_0_0.txt", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    try:
                        addr = int(parts[0].strip())
                        val = int(parts[1].strip())
                        mem_map[addr] = val
                    except: pass
        
        # Convert map to list based on test expectations
        # Most tests expect data at 2560 (1024+1024+512)
        base = 2560
        vals = []
        for i in range(128):
            vals.append(mem_map.get(base + i*4, 0))
        return vals

    # Fallback to parsing stdout (legacy)
    lines = output.split('\n')
    memory_dump = []
    in_dump = False
    for line in lines:
        if "Memory Dump" in line:
            in_dump = True
            continue
        if in_dump and line.startswith("Row"):
            parts = line.split(':')[1].strip().split()
            memory_dump.extend([int(x) for x in parts])
    return memory_dump

def load_gold(filename):
    gold_vals = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("Row"):
                parts = line.split(':')[1].strip().split()
                gold_vals.extend([int(x) for x in parts])
    return gold_vals

def test_debug():
    print("\n--- Testing Debug Kernel (Gid) ---")
    success, _ = run_cmd(_py("apps/app_cuda_debug.py"))
    if not success: return False
    success, output = run_cmd("cd sim && make clean && make run")
    if not success: return False
    
    sim_vals = parse_simulation_output(output)
    # Expected: 0..127
    errors = 0
    for i in range(128):
        if i < len(sim_vals) and sim_vals[i] != i:
            print(f"Mismatch at {i}: Exp {i}, Got {sim_vals[i]}")
            errors += 1
            if errors > 10: break
    
    if errors == 0 and len(sim_vals) >= 128:
        print("[PASS] Debug Test Passed")
        return True
    return False

def test_load():
    print("\n--- Testing Load Kernel (Copy A to C) ---")
    success, _ = run_cmd(_py("apps/app_cuda_load_test.py"))
    if not success: return False
    success, output = run_cmd("cd sim && make clean && make run")
    if not success: return False
    
    sim_vals = parse_simulation_output(output)
    # Expected: 0..127 (since A was init to 0..127)
    errors = 0
    for i in range(128):
        if i < len(sim_vals) and sim_vals[i] != i:
            print(f"Mismatch at {i}: Exp {i}, Got {sim_vals[i]}")
            errors += 1
            if errors > 10: break
            
    if errors == 0 and len(sim_vals) >= 128:
        print("[PASS] Load Test Passed")
        return True
    return False

def test_vecmul():
    print("\n--- Testing VecMul Kernel (2 * 3 = 6) ---")
    success, _ = run_cmd(_py("apps/app_cuda_vecmul_test.py"))
    if not success: return False
    success, output = run_cmd("cd sim && make clean && make run")
    if not success: return False
    
    sim_vals = parse_simulation_output(output)
    # Expected: 6
    errors = 0
    for i in range(128):
        if i < len(sim_vals) and sim_vals[i] != 6:
            print(f"Mismatch at {i}: Exp 6, Got {sim_vals[i]}")
            errors += 1
            if errors > 10: break
    
    if errors == 0 and len(sim_vals) >= 128:
        print("[PASS] VecMul Test Passed")
        return True
    return False

def test_matmul():
    print("\n--- Testing Matrix Multiplication (Random Data) ---")
    # 1. Run App Generator
    success, _ = run_cmd(_py("apps/app_cuda_matmul.py"))
    if not success: return False

    # 2. Run Simulation
    success, output = run_cmd("cd sim && make clean && make run")
    if not success: return False

    # 3. Verify
    sim_vals = parse_simulation_output(output)
    gold_vals = load_gold("tests/gold.txt")

    # Compare
    errors = 0
    # We check first 128 elements (16x8 matrix)
    check_len = 16 * 8 
    
    for i in range(min(len(sim_vals), len(gold_vals), check_len)):
        if sim_vals[i] != gold_vals[i]:
            print(f"Mismatch at index {i}: Expected {gold_vals[i]}, Got {sim_vals[i]}")
            errors += 1
            if errors > 20:
                print("... Too many errors ...")
                break
    
    if errors == 0:
        print("[PASS] MatMul Test Passed!")
        return True
    else:
        print(f"[FAIL] MatMul Test Failed with {errors} errors.")
        return False

def test_loop():
    print("\n--- Testing Loop Kernel (acc += 1 for 16 times) ---")
    success, _ = run_cmd(_py("apps/app_cuda_loop_test.py"))
    if not success: return False
    success, output = run_cmd("cd sim && make clean && make run")
    if not success: return False
    
    sim_vals = parse_simulation_output(output)
    # Expected: 16
    errors = 0
    for i in range(128):
        if i < len(sim_vals) and sim_vals[i] != 16:
            print(f"Mismatch at {i}: Exp 16, Got {sim_vals[i]}")
            errors += 1
            if errors > 10: break
    
    if errors == 0 and len(sim_vals) >= 128:
        print("[PASS] Loop Test Passed")
        return True
    return False

def test_divergence():
    print("\n--- Testing Divergence Kernel (Odd=11, Even=12) ---")
    success, _ = run_cmd(_py("apps/app_cuda_divergence.py"))
    if not success: return False
    success, output = run_cmd("cd sim && make clean && make run")
    if not success: return False
    
    sim_vals = parse_simulation_output(output)
    errors = 0
    for i in range(128):
        expected = 11 if (i % 2 != 0) else 12
        if i < len(sim_vals) and sim_vals[i] != expected:
            print(f"Mismatch at {i}: Exp {expected}, Got {sim_vals[i]}")
            errors += 1
            if errors > 10: break
            
    if errors == 0 and len(sim_vals) >= 128:
        print("[PASS] Divergence Test Passed")
        return True
    return False

def test_fp4():
    """FP4 E2M1 乘法与 rtl/fp4_unit 对齐（见 apps/app_fp4_test.py）。"""
    print("\n--- Testing FP4 E2M1 Multiply ---")
    success, _ = run_cmd(_py("apps/app_fp4_test.py"))
    if not success:
        return False
    success, output = run_cmd("cd sim && make clean && make run")
    if not success:
        return False

    gold_path = "tests/fp4_gold.txt"
    if not os.path.exists(gold_path):
        print(f"[FAIL] Missing {gold_path}")
        return False
    gold_vals = []
    with open(gold_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                gold_vals.append(int(line))

    sim_vals = parse_simulation_output(output)
    errors = 0
    n = min(len(gold_vals), len(sim_vals), 128)
    for i in range(n):
        if sim_vals[i] != gold_vals[i]:
            print(f"Mismatch at {i}: Exp {gold_vals[i]}, Got {sim_vals[i]}")
            errors += 1
            if errors > 10:
                break
    if errors == 0 and n >= 128:
        print("[PASS] FP4 Test Passed")
        return True
    print(f"[FAIL] FP4 Test Failed with {errors} errors.")
    return False


def test_fc():
    print("\n--- Testing Fully Connected Layer ---")
    # 1. Run App Generator
    success, _ = run_cmd(_py("apps/app_cuda_fc.py"))
    if not success: return False

    # 2. Run Simulation
    success, output = run_cmd("cd sim && make clean && make run")
    if not success: return False

    # 3. Verify
    sim_vals = parse_simulation_output(output)
    
    # Expected Logic:
    # Rows 0-7: 16 (16 inputs * 1 weight + 0 bias)
    # Rows 8-15: 0 (16 inputs * -2 weight + 10 bias = -22 -> ReLU -> 0)
    
    errors = 0
    for i in range(16): # Check first 16 outputs
        expected = 16 if i < 8 else 0
        if i < len(sim_vals) and sim_vals[i] != expected:
             print(f"Mismatch at {i}: Exp {expected}, Got {sim_vals[i]}")
             errors += 1
             if errors > 10: break
    
    if errors == 0 and len(sim_vals) >= 16:
        print("[PASS] FC Test Passed")
        return True
    else:
        print(f"[FAIL] FC Test Failed with {errors} errors.")
        return False

def main():
    tests = [
        test_debug,
        test_load,
        test_loop,
        test_divergence,
        test_vecmul,
        test_matmul,
        test_fc,
        test_fp4,
    ]
    
    passed = 0
    for t in tests:
        if t(): passed += 1
        
    print(f"\nSummary: {passed}/{len(tests)} Tests Passed")
    if passed != len(tests):
        sys.exit(1)

if __name__ == "__main__":
    main()
