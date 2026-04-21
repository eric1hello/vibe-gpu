# -*- coding: utf-8 -*-
"""
FP4（E2M1）验证：与 rtl/fp4_unit.sv / tools/fp4_soft.py 对齐。
使用 CudaCompiler.emit_hw_id 与 load_large_const，避免手写立即数越界或寄存器冲突。
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tools.assembler import Assembler
from tools.fp4_soft import mul_fp4
from tools.vibe_cuda import CudaCompiler


def main():
    pat_a = [0x2, 0x3, 0x4, 0x6]
    pat_b = [0x2, 0x2, 0x3, 0x5]
    vals_a = [pat_a[i % 4] for i in range(128)]
    vals_b = [pat_b[i % 4] for i in range(128)]

    expected_mul = [mul_fp4(vals_a[i], vals_b[i]) for i in range(128)]

    base_a = 1024
    base_b = 1152
    base_out = 2560

    asm = Assembler(base_data_addr=base_a)
    cc = CudaCompiler(asm, {}, 32)

    cc.emit_hw_id(1)

    asm.ldi(2, base_a)
    asm.ldi(4, 2)
    asm.sll(3, 1, 4)
    asm.mov(14, 3)

    asm.add(4, 2, 3)
    asm.ldw(5, 4)

    asm.ldi(6, base_b)
    asm.add(7, 6, 14)
    asm.ldw(8, 7)

    asm.fmul4(9, 5, 8)

    cc.load_large_const(10, base_out)
    asm.add(11, 10, 14)
    asm.stw(11, 9)
    asm.halt()

    asm.base_data_addr = base_a
    asm.data_offset = 0
    asm.data(vals_a)
    asm.data(vals_b)

    root = os.path.join(os.path.dirname(__file__), "..")
    asm.write_hex(os.path.join(root, "tests", "program.hex"))

    gold_path = os.path.join(root, "tests", "fp4_gold.txt")
    with open(gold_path, "w", encoding="utf-8") as f:
        for i, v in enumerate(expected_mul):
            f.write(f"{v}\n")

    print("FP4 测试已生成：tests/program.hex, tests/fp4_gold.txt")


if __name__ == "__main__":
    main()
