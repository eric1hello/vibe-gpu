# 无访存：仅用常数验证 FMUL4 + 写回（HW_ID=0 路径简化）
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tools.assembler import Assembler

def main():
    asm = Assembler(base_data_addr=1024)
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
    print("synth fmul4 2*3 fp4 -> mem[2560]")

if __name__ == "__main__":
    main()
