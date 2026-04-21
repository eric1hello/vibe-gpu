# HW_ID 计算后接与 app_fp4_synth 相同的常数 FMUL4+STW
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tools.assembler import Assembler

def main():
    asm = Assembler(base_data_addr=1024)
    asm.warpid(1)
    asm.ldi(2, 3)
    asm.sll(1, 1, 2)
    asm.tid(2)
    asm.add(1, 1, 2)
    asm.smid(2)
    asm.ldi(3, 5)
    asm.sll(2, 2, 3)
    asm.add(1, 1, 2)
    asm.ldi(5, 2)
    asm.ldi(8, 3)
    asm.fmul4(9, 5, 8)
    asm.ldi(10, 1024)
    asm.ldi(12, 1024)
    asm.add(10, 10, 12)
    asm.ldi(12, 512)
    asm.add(10, 10, 12)
    asm.stw(10, 9)
    asm.halt()
    root = os.path.join(os.path.dirname(__file__), "..")
    asm.write_hex(os.path.join(root, "tests", "program.hex"))
    print("hw+synth")

if __name__ == "__main__":
    main()
