# -*- coding: utf-8 -*-
"""仅运行 Tensor Core TCDP4 测试（需已安装 Python 与 Verilator 仿真环境）。"""
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(ROOT)


def main():
    py = sys.executable
    r = subprocess.run([py, "apps/app_tensor_test.py"], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)
    r = subprocess.run("cd sim && make clean && make run", shell=True, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)

    dump = os.path.join(ROOT, "sim", "dump_mem_0_0.txt")
    gold_path = os.path.join(ROOT, "tests", "tensor_gold.txt")
    if not os.path.isfile(dump):
        print("[FAIL] 缺少 sim/dump_mem_0_0.txt")
        sys.exit(1)

    mem = {}
    with open(dump, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) == 2:
                try:
                    mem[int(parts[0].strip())] = int(parts[1].strip())
                except ValueError:
                    pass

    base = 2560
    with open(gold_path, "r", encoding="utf-8") as f:
        gold = [int(x.strip()) for x in f if x.strip()]

    err = 0
    for i in range(min(128, len(gold))):
        got = mem.get(base + i * 4, -1)
        if got != gold[i]:
            print(f"[FAIL] i={i} 期望 {gold[i]} 得到 {got}")
            err += 1
            if err > 8:
                break

    if err:
        sys.exit(1)
    print("[PASS] Tensor Core (TCDP4) 测试通过。波形: sim/sim_trace.vcd")


if __name__ == "__main__":
    main()
