# Vibe GPU 架构与文档（中文）

本文档从硬件与软件两方面说明 **Vibe GPU** 的设计。本仓库在 [PKUZHOU/Vibe-GPU](https://github.com/PKUZHOU/Vibe-GPU) 基础上演进，增加了 **FP4（E2M1）** 标量指令与 **Tensor Core 类比** 的规约指令 `TCDP4`。

## 1. 硬件架构

Vibe GPU 为教学与研究用途的自定义 **SIMT GPGPU**，采用 32 位类 RISC ISA，支持大规模并行、控制流分歧与浮点运算。

![硬件架构](images/arch_hardware.svg)

### 1.1 顶层结构

- **SM 核心**：共 4 个流式多处理器（Streaming Multiprocessor）。
- **存储模型**：统一地址空间。
  - **主存**：仿真中为多核共享的字节可寻址空间（如 4KB 级）。
  - **L1 缓存**：每个 SM 自带 L1 数据缓存以降低访存延迟。
  - **互连**：简单仲裁器将多路 SM 接到共享主存。

### 1.2 流式多处理器（SM）

每个 SM 管理多个 **Warp**（线程束），以流水线方式执行。

#### 规格

- **每 Warp 线程数**：8  
- **每 SM Warp 数**：4  
- **每 SM 线程总数**：32  
- **全 GPU 活跃线程**：128（4×32）  
- **寄存器**：每线程 32 个 32 位通用寄存器  

#### 流水线阶段

五级流水线：

1. **IF（取指）**  
   - **Warp 调度器**：在活跃、未 halt 的 Warp 间 **轮询** 选择。  
   - 从当前 Warp 的 PC 取指，并更新 PC。

2. **ID（译码）**  
   - 译码 32 位指令。  
   - **冒险检测**：通过 **记分板** 检测 RAW 等数据相关。  
   - **读寄存器**：从线程私有寄存器堆读 `rs1`、`rs2`。  
   - **结构冒险**：若上一条访存未完成，可插入停顿。

3. **EX（执行）**  
   - **ALU**：整数 `ADD/SUB/MUL/AND/OR/SLL/SRL` 等。  
   - **FP8**：`FADD`/`FMUL`，操作数为寄存器 **低 8 位**（E4M3）。  
   - **CUDA Core 类 FP4**：`FADD4`/`FMUL4`，每线程对 **两个寄存器各取低 4 位** 做标量二元运算（E2M1）。  
   - **Tensor Core 类**：`TCDP4`，对 `rs1`/`rs2` **低 16 位** 中打包的 **4 路 FP4** 做点积规约（见 1.6 节）。  
   - **分支**：`BEQ`/`BNE` 条件判断。  
   - **SIMT 栈**：处理控制流分歧（见 1.3 节）。

4. **MEM（访存）**  
   - `LDW`/`STW` 与 **L1 缓存** 交互。  
   - 缓存未命中时流水线 **停顿**，直到数据就绪。

5. **WB（写回）**  
   - 将结果写回寄存器堆，并 **清除记分板** 对应项。

### 1.3 SIMT 分歧处理

当 `if/else` 等导致 Warp 内线程走不同路径时，需要硬件支持 **分歧与汇合**。

- **SIMT 栈**：每个 Warp 维护栈，记录执行掩码与返回 PC。  
- **执行掩码**：标明当前哪些线程有效。  
- **机制概要**：  
  1. **分歧**：部分线程跳转、部分不跳时，硬件将 **未采纳路径** 的掩码与 PC **压栈**，并只执行采纳分支的线程。  
  2. **汇合**：`JOIN` 从栈中 **弹出**，恢复掩码与 PC，使线程重新汇合。

### 1.4 存储与冒险

- **记分板**：目的寄存器有未完成写回时阻塞相关指令发射。  
- **全局停顿**：访存未命中等情况下插入气泡或停顿整级流水线。  
- **L1 握手**：通过请求/完成握手保证访存在停顿多周期时仍语义正确。

### 1.5 FP4（E2M1）数据通路 — CUDA Core 类标量

- **格式**：`S(1) | E(2) | M(1)`，指数偏置 **1**；`E=0` 按 **零** 处理（非规格 flush）。  
- **正规数**：  
  \(\;(-1)^S \cdot 2^{(E-1)} \cdot (1 + M/2)\;\)，\(E \in \{1,2,3\}\)。  
- **实现**：`rtl/fp4_unit.sv` 内将 FP4 与 **定点（缩放因子 FIX=256）** 互转；与 `tools/fp4_soft.py` 对齐。  
- **指令**：`FADD4`（`0x0D`）、`FMUL4`（`0x0E`），**每线程一次只处理一对 FP4 标量**（取自 `rs1`/`rs2` 低 4 位），类比 GPU 上 **CUDA Core** 对 FP32/FP16 的逐线程标量运算。

### 1.6 CUDA Core 与 Tensor Core（教学类比）

商用 GPU 中，**CUDA Core** 负责通用标量/向量浮点与整数；**Tensor Core** 负责矩阵乘加片段（如 WMMA）的高吞吐规约。本设计用 **两条可区分的硬件路径** 做极简对应：

| 类比 | 硬件 | 指令 | 数据形态 |
|------|------|------|----------|
| CUDA Core 类 | `fp8_unit`、`fp4_unit` | `FADD`/`FMUL`、`FADD4`/`FMUL4` | 每线程 **标量**（低 8/4 位） |
| Tensor Core 类 | `tensor_core.sv` | `TCDP4`（`0x0F`） | `rs1`/`rs2` **低 16 位** 各含 **4×FP4**，做点积 \(\sum_{i=0}^{3} a_i b_i\) 后编码为一个 FP4 写入 `rd` |

`TCDP4` **不是**完整矩阵乘，而是 **4-lane 点积片段**，便于在 RTL/文档中把「规约型」运算与「标量二元」运算分开。参考模型：`tools/fp4_soft.dot4_fp4`。

## 2. 软件架构

从高层 Python 到机器码的完整工具链如下。

![软件栈](images/arch_software.svg)

### 2.1 第二层：Vibe CUDA 前端（`tools/vibe_cuda.py`）

- 使用 `@cuda.jit` 标记内核。  
- 内置 `blockIdx`、`threadIdx`、`blockDim` 等概念（由编译器映射到 `TID`/`WARPID`/`SMID` 等）。  
- **AST 分析**：解析 `if`、`for`、赋值等。  
- **寄存器分配**：简单线性分配，映射到 `R1`–`R31`。  
- **控制流**：自动生成 `BEQ`/`BNE`/`JOIN` 等。

### 2.2 第一层：汇编器（`tools/assembler.py`）

- 将汇编助记符编码为 32 位指令，输出 **`tests/program.hex`**，由仿真顶层 `$readmemh` 载入。

### 2.3 寄存器与指令摘要

- **`R0`**：恒 0。  
- **`R1`–`R31`**：通用寄存器（每线程私有）。  

完整操作码表见 **[isa.md](isa.md)**（与 `tools/isa.py`、`rtl/defines.svh` 一致）。

## 3. 目录结构

```
.
├── apps/                 # 应用内核（Python）
├── docs/                 # 文档与图片（本文件、isa.md）
├── rtl/                  # SystemVerilog
│   ├── sm_core.sv        # SM 与流水线
│   ├── alu.sv            # ALU + FP8 + FP4
│   ├── fp4_unit.sv       # FP4 标量（CUDA Core 类）
│   ├── tensor_core.sv    # FP4 四路点积（Tensor Core 类）
│   ├── fp8_unit.sv
│   ├── simt_stack.sv
│   └── ...
├── sim/                  # Verilator 仿真
├── tests/                # program.hex、黄金值、test_suite.py、test_fp4_only.py
└── tools/                # vibe_cuda、assembler、isa、fp4_soft
```

## 4. 验证与波形

- 生成 FP4 程序与黄金值：`python apps/app_fp4_test.py`  
- 仅测 FP4：`python tests/test_fp4_only.py`  
- 全量：`python tests/test_suite.py`  
- 仿真：`cd sim && make run`  
- **波形文件**：`sim/sim_trace.vcd`，可用 **GTKWave** 打开，观察 PC、流水线寄存器、ALU 结果等。

## 5. 与上游仓库的关系

本设计继承 [Vibe-GPU](https://github.com/PKUZHOU/Vibe-GPU) 的整体 SIMT 与工具链结构，在保持兼容的前提下扩展 FP4 数据通路与指令；架构说明与 ISA 说明均以 **中文** 维护于本目录。
