# HW_ID（与 emit_hw_id 一致）+ 地址 + LDW + 写回
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tools.assembler import Assembler

def main():
    asm = Assembler(base_data_addr=1024)
    asm.warpid(1)
    asm.ldi(14, 3)
    asm.sll(1, 1, 14)
    asm.tid(14)
    asm.add(1, 1, 14)
    asm.smid(14)
    asm.ldi(15, 5)
    asm.sll(14, 14, 15)
    asm.add(1, 1, 14)

    asm.ldi(2, 1024)
    asm.ldi(4, 2)
    asm.sll(3, 1, 4)
    asm.add(4, 2, 3)
    asm.ldw(5, 4)

    asm.ldi(10, 1024)
    asm.ldi(12, 1024)
    asm.add(10, 10, 12)
    asm.ldi(12, 512)
    asm.add(10, 10, 12)
    asm.stw(10, 5)
    asm.halt()

    asm.base_data_addr = 1024
    asm.data_offset = 0
    asm.data([2] * 128)
    root = os.path.join(os.path.dirname(__file__), "..")
    asm.write_hex(os.path.join(root, "tests", "program.hex"))
    print("ldw only v2")

if __name__ == "__main__":
    main()
