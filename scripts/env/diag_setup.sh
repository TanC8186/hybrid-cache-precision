#!/usr/bin/env bash
# 定位 vLLM setup.py metadata 挂点
set -uo pipefail
LOG=/root/setup_diag.log
cd /root/MLSys_Research/vendor/vllm
source /root/MLSys_Research/.venv/bin/activate
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.8.4.dev
export VLLM_VERSION_OVERRIDE=0.8.4.dev
export TORCH_CUDA_ARCH_LIST=12.0
export CUDA_HOME=/usr/local/cuda
export GIT_TERMINAL_PROMPT=0

# 后台跑 setup.py，看它输出到哪一步
python setup.py --version > "$LOG" 2>&1 &
SPID=$!
sleep 25
echo "=== 25s 后 setup.py 输出 ===" > /root/setup_status.txt
cat "$LOG" >> /root/setup_status.txt 2>/dev/null
echo "=== 进程状态 ===" >> /root/setup_status.txt
ps -o pid,stat,wchan:25,cmd -p $SPID 2>/dev/null >> /root/setup_status.txt
echo "=== 子进程 ===" >> /root/setup_status.txt
ps --ppid $SPID -o pid,stat,wchan:20,cmd 2>/dev/null | head -6 >> /root/setup_status.txt
# 杀掉
kill $SPID 2>/dev/null
echo "done" >> /root/setup_status.txt
