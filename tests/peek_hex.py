import sys
sys.path.insert(0, ".")
from tools.isa import *

def main():
    lines = open("tests/program.hex").read().split()
    words = []
    for i in range(0, len(lines), 4):
        if i + 3 >= len(lines):
            break
        b = bytes(int(lines[j], 16) for j in range(i, i + 4))
        words.append(int.from_bytes(b, "little"))
    for idx, w in enumerate(words[:45]):
        op = (w >> 26) & 0x3F
        print(idx, hex(w), "op", hex(op), "rd", (w >> 21) & 0x1F)
    print("total words", len(words))
    print("--- last 8 inst before padding: find HALT 3f ---")
    for i, w in enumerate(words):
        if ((w >> 26) & 0x3F) == 0x3F:
            print("HALT at word index", i, hex(w))

if __name__ == "__main__":
    main()
