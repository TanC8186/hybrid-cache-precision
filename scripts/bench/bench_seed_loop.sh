#!/bin/bash
# usage: bench_seed_loop.sh <model> <result_dir> <log_prefix> <seed1,seed2,seed3> <rates...>
MODEL="$1"; RESULT_DIR="$2"; LOGPREFIX="$3"; SEEDS="$4"; shift 4
for S in $(echo "$SEEDS" | tr ',' ' '); do
  /root/autodl-tmp/bench_driver_5090.sh "$MODEL" "$RESULT_DIR" "$S" "${LOGPREFIX}_seed${S}.log" "$@"
done
echo "### ALL SEEDS DONE"
