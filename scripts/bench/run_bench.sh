#!/bin/bash
# usage: run_bench.sh <rate> <result_dir> <base_url> [timeout_sec]
RATE=$1
DIR=$2
URL=$3
TO=${4:-300}
MODEL=/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master
mkdir -p $DIR
cd /root/autodl-tmp/MLSys_Research
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
timeout $TO vllm bench serve --backend openai --base-url $URL --model "$MODEL" --dataset-name random --num-prompts 400 --request-rate $RATE --max-concurrency 512 --save-result --result-dir $DIR --random-input-len 1024 --random-output-len 128 > $DIR/rate_${RATE}.log 2>&1
echo "BENCH_EXIT=$?" >> $DIR/rate_${RATE}.log
