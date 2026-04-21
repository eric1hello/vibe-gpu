# 最小 STW 测试：向 2560 写入常数 7
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tools.assembler import Assembler

def main():
    asm = Assembler(base_data_addr=1024)
    asm.ldi(5, 7)
    asm.ldi(10, 1024)
    asm.ldi(12, 1024)
    asm.add(10, 10, 12)
    asm.ldi(12, 512)
    asm.add(10, 10, 12)
    asm.stw(10, 5)
    asm.halt()
    root = os.path.join(os.path.dirname(__file__), "..")
    asm.write_hex(os.path.join(root, "tests", "program.hex"))
    print("min store -> 2560 = 7")

if __name__ == "__main__":
    main()
