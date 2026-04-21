# Vibe GPU 自定义 SIMT 指令集（ISA）

本文档描述 Vibe GPU 的指令集架构。体系为 **32 位**、类 RISC、**SIMT** 执行模型。

## 架构概览

- **数据宽度**：32 位  
- **指令宽度**：32 位  
- **每 Warp 线程数**：8  
- **SM 数量**：4（可配置）  
- **寄存器**：每线程 32 个通用寄存器（`R0`–`R31`），`R0` 恒为 0  

## 指令格式

所有指令定长 32 位：

| 位域 | 宽度 | 说明 |
|------|------|------|
| `[31:26]` | 6 | 操作码 Opcode |
| `[25:21]` | 5 | 目的寄存器 Rd |
| `[20:16]` | 5 | 源寄存器 1 Rs1 |
| `[15:11]` | 5 | 源寄存器 2 Rs2 |
| `[10:0]` | 11 | 立即数（有符号扩展） |

## 操作码一览

| 操作码 | 助记符 | 格式 | 说明 |
|--------|--------|------|------|
| `0x00` | NOP | — | 空操作 |
| `0x01` | ADD | R,R,R | `Rd = Rs1 + Rs2` |
| `0x02` | SUB | R,R,R | `Rd = Rs1 - Rs2` |
| `0x03` | LDI | R,Imm | `Rd = SignExt(Imm)`，立即数为 11 位 |
| `0x04` | AND | R,R,R | 按位与 |
| `0x05` | OR | R,R,R | 按位或 |
| `0x06` | MOV | R,R | `Rd = Rs1` |
| `0x07` | MUL | R,R,R | 整数乘法 |
| `0x08` | FADD | R,R,R | FP8 加（操作数低 8 位为 E4M3） |
| `0x09` | FMUL | R,R,R | FP8 乘 |
| `0x0A` | CSR | R,Rs1 | 读 CSR（由 Rs1 选择索引） |
| `0x0B` | SLL | R,R,R | 逻辑左移 |
| `0x0C` | SRL | R,R,R | 逻辑右移 |
| `0x0D` | FADD4 | R,R,R | FP4 加（操作数**低 4 位**为 E2M1） |
| `0x0E` | FMUL4 | R,R,R | FP4 乘（标量，CUDA Core 类） |
| `0x0F` | TCDP4 | R,R,R | Tensor Core 类：低 16 位 4×FP4 点积 |
| `0x10` | LDW | R,Rs1 | 从 `[Rs1+Imm]` 加载字（寻址在译码阶段与立即数组合，见实现） |
| `0x11` | STW | Rs1,Rs2 | 存储字 |
| `0x12` | TID | Rd | `Rd` = 线程 ID（Warp 内 0–7） |
| `0x13` | SMID | Rd | `Rd` = SM 编号 |
| `0x14` | WARPID | Rd | `Rd` = Warp 编号 |
| `0x20` | BEQ | Rs1,Rs2,Imm | 相等则分支 |
| `0x21` | BNE | Rs1,Rs2,Imm | 不等则分支 |
| `0x22` | JOIN | — | SIMT 栈汇合 |
| `0x3F` | HALT | — | 停止当前 Warp |

> 存储器访问的具体寻址方式以 `rtl/sm_core.sv` 与汇编器为准；上表为语义概要。

## SIMT 行为

- 每条指令在 **Warp 内** 被多个线程并行执行（由执行掩码控制活跃线程）。
- `TID`、`WARPID`、`SMID` 用于计算全局线性线程 ID 与访存偏移。
- 分支分歧由 **SIMT 栈** 与 `JOIN` 配合硬件完成（详见 `docs/ARCHITECTURE.md`）。

## FP4（E2M1）补充

- `FADD4` / `FMUL4` 仅使用源寄存器的 **低 4 位** 作为 FP4 数据；结果写入 `Rd` 时同样以低 4 位为有效浮点位，高位零扩展。
- 格式定义与定点约定见 `docs/ARCHITECTURE.md` 与 `tools/fp4_soft.py`。
