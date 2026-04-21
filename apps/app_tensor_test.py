# -*- coding: utf-8 -*-
"""
Tensor Core（TCDP4）仿真用例：与 rtl/tensor_core.sv、tools/fp4_soft.dot4_fp4 对齐。
rs1/rs2 低 16 位各含 4×FP4 nibble；结果写入 2560 + hw_id*4。
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tools.assembler import Assembler
from tools.fp4_soft import dot4_fp4
from tools.vibe_cuda import CudaCompiler


def main():
    # 每线程一对 32 位字，仅低 16 位参与点积（高 16 位可为 0）
    pat_a = [0x2222, 0x3232, 0x4544, 0x5656]
    pat_b = [0x2323, 0x3434, 0x5555, 0x6767]
    vals_a = [pat_a[i % 4] for i in range(128)]
    vals_b = [pat_b[i % 4] for i in range(128)]

    expected = [dot4_fp4(vals_a[i], vals_b[i]) for i in range(128)]

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

    asm.tcdp4(9, 5, 8)

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

    gold_path = os.path.join(root, "tests", "tensor_gold.txt")
    with open(gold_path, "w", encoding="utf-8") as f:
        for v in expected:
            f.write(f"{v}\n")

    print("Tensor Core (TCDP4) 测试已生成：tests/program.hex, tests/tensor_gold.txt")


if __name__ == "__main__":
    main()
