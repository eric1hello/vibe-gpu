# -*- coding: utf-8 -*-
"""
FP4 (E2M1) 软件参考模型 —— 与 rtl/fp4_unit.sv 中定点约定一致，用于测试与黄金值生成。
格式：S(1) E(2) M(1)，偏置 bias=1；E=0 视为零（非规格 flushed）。
数值：(-1)^S * 2^(E-1) * (1 + M/2)，其中 E ∈ {1,2,3}。
"""

FIX = 256  # 与 RTL 中定点缩放一致（2^8）


def decode_fp4(x: int) -> int:
    """有符号定点值，缩放 FIX（与硬件 mag_fix 一致）。"""
    x &= 0xF
    if (x & 0xE) == 0:
        return 0
    s = (x >> 3) & 1
    e = (x >> 1) & 3
    m = x & 1
    if e == 0:
        return 0
    # mag = 2^(e-1) * (1 + m/2) * FIX
    mag = ((1 << (e - 1)) * (2 + m) * FIX) >> 1
    return -mag if s else mag


def encode_fp4(val_fix: int) -> int:
    """从定点有符号值编码为 FP4 nibble。"""
    if val_fix == 0:
        return 0
    neg = val_fix < 0
    absv = -val_fix if neg else val_fix
    best_e, best_m, best_err = 1, 0, 1 << 30
    for e in range(1, 4):
        for m in range(2):
            tgt = ((1 << (e - 1)) * (2 + m) * FIX) >> 1
            err = abs(absv - tgt)
            if err < best_err:
                best_err = err
                best_e, best_m = e, m
    s = 1 if neg else 0
    return s << 3 | best_e << 1 | best_m


def mul_fp4(a: int, b: int) -> int:
    va = decode_fp4(a)
    vb = decode_fp4(b)
    if va == 0 or vb == 0:
        return 0
    # 与 Verilog 有符号除法一致：向 0 截断
    p = va * vb
    if p >= 0:
        prod = p // FIX
    else:
        prod = -(-p // FIX)
    return encode_fp4(prod)


def add_fp4(a: int, b: int) -> int:
    return encode_fp4(decode_fp4(a) + decode_fp4(b))
