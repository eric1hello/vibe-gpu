#!/usr/bin/env bash
# 在 WSL/Ubuntu 中运行 FP4 仿真并生成波形（需已安装：verilator make g++）
# 用法：在 WSL 里执行：bash scripts/wsl_sim_fp4.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/bin:/usr/bin:$PATH"
if ! command -v verilator >/dev/null 2>&1; then
  echo "未找到 verilator。请先执行：sudo apt-get update && sudo apt-get install -y verilator make g++"
  exit 1
fi
PYTHON="${PYTHON:-python3}"
"$PYTHON" apps/app_fp4_test.py
(cd sim && make clean && make run)
echo "完成。波形: $ROOT/sim/sim_trace.vcd"
echo "可用 GTKWave 打开，或：gtkwave sim/sim_trace.vcd &"
