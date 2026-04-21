# -*- coding: utf-8 -*-
"""仅运行 FP4 乘法测试（不跑完整 test_suite）。需已安装 Python。"""
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(ROOT)


def main():
    py = sys.executable
    r = subprocess.run([py, "apps/app_fp4_test.py"], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)
    r = subprocess.run("cd sim && make clean && make run", shell=True, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)
    # 校验 dump
    dump = os.path.join(ROOT, "sim", "dump_mem_0_0.txt")
    gold_path = os.path.join(ROOT, "tests", "fp4_gold.txt")
    if not os.path.isfile(dump):
        print("[FAIL] 缺少 sim/dump_mem_0_0.txt，请确认仿真已结束。")
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
    print("[PASS] FP4 测试通过。波形: sim/sim_trace.vcd")
    vcd = os.path.join(ROOT, "sim", "sim_trace.vcd")
    if os.path.isfile(vcd):
        print(f"可用 GTKWave 打开: {vcd}")


if __name__ == "__main__":
    main()
