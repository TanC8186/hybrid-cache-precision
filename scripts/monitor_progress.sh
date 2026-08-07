#!/usr/bin/env bash
# Compact progress monitor for the 2026-08-07 chained GPU pipeline.
cd /root/autodl-tmp/MLSys_Research || exit 1

niah_count=$(ls results/quality/niah-fixed/niah-fixed-20260807/*.json 2>/dev/null | wc -l)
ruler_count=$(ls results/quality/ruler-subset/ruler-subset-20260807/*.json 2>/dev/null | wc -l)
ruler9b_count=$(ls results/quality/niah-fixed-9b/niah-fixed-9b-20260807/*.json 2>/dev/null | wc -l)
reasoning_count=$(ls results/quality/reasoning/reasoning-20260807/*.json 2>/dev/null | wc -l)

echo "== stages =="
echo "NIAH fixed:   $niah_count / 90"
echo "RULER subset: $ruler_count / 70"
echo "NIAH 9B:      $ruler9b_count / 54"
echo "Reasoning:    $reasoning_count / 15"
if grep -q '\[DONE\] niah-fixed-20260807' logs/niah-fixed-20260807.log 2>/dev/null; then echo "NIAH: DONE"; fi
if grep -q '\[DONE\] ruler-subset-20260807' logs/ruler-subset-20260807.log 2>/dev/null; then echo "RULER: DONE"; fi
if grep -q '\[DONE_GATES\]' logs/r5-serving-v3-gates-20260807.log 2>/dev/null; then echo "SERVING_GATES: DONE"; fi
if grep -q '\[DONE\] niah-fixed-9b-20260807' logs/niah-fixed-9b-20260807.log 2>/dev/null; then echo "NIAH_9B: DONE"; fi
if grep -q '\[DONE\] reasoning-20260807' logs/reasoning-20260807.log 2>/dev/null; then echo "REASONING: DONE"; fi

echo "== serving gates log tail =="
tail -2 logs/r5-serving-v3-gates-20260807.log 2>/dev/null
echo "== gpu =="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
date +%H:%M:%S
