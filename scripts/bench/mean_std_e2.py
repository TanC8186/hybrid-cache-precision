#!/usr/bin/env python3
"""Compute mean ± std across seeds for vllm bench result JSONs, grouped by request_rate.
Usage: mean_std_e2.py <dir> [label]
Expected filenames like openai-<rate>qps-concurrency512-master-*.json; seed read from metadata (or filename).
Outputs one row per rate: mean±std of req_tp, out_tok/s, ttft_p99, tpot_p50, tpot_p99.
"""
import json, glob, os, sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(os.path.normpath(d))

files = sorted(glob.glob(os.path.join(d, "*.json")))
if not files:
    print(f"NO_JSON {d}")
    sys.exit(0)

from collections import defaultdict
groups = defaultdict(list)
for f in files:
    j = json.load(open(f))
    r = j["request_rate"]
    seed = j.get("seed", "?")
    groups[r].append((seed, j))

rates = sorted(groups)
print(f"## {label}  ({len(files)} files, {len(rates)} rates)")
print("rate  n  req_tp(mean±std)      out_tok/s(mean±std)   ttft_p99(mean±std)  tpot_p50(mean±std)  tpot_p99(mean±std)  seeds")
for r in rates:
    items = groups[r]
    n = len(items)
    def ms(key):
        vals = [it[key] for _, it in items]
        m = sum(vals) / n
        sd = (sum((v - m) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
        return m, sd
    rt, rts = ms("request_throughput")
    ot, ots = ms("output_throughput")
    t99, t99s = ms("p99_ttft_ms")
    tp50, tp50s = ms("median_tpot_ms")
    tpp99, tpp99s = ms("p99_tpot_ms")
    seeds = ",".join(s for s, _ in items)
    print(f"{r:>5.0f} {n:2d} {rt:6.2f}±{rts:4.2f}  {ot:8.1f}±{ots:6.1f}   {t99:8.1f}±{t99s:7.1f}  {tp50:6.2f}±{tp50s:4.2f}  {tpp99:7.2f}±{tpp99s:6.2f}  {seeds}")

# E3 SLO: max rate with ttft_p99 mean < 2000ms
print("\nE3 SLO (TTFT p99 < 2000ms):")
for r in rates:
    items = groups[r]
    t99vals = [it["p99_ttft_ms"] for _, it in items]
    m = sum(t99vals) / len(t99vals)
    flag = "OK" if m < 2000 else "VIOLATE"
    print(f"  rate {r:>5.0f}: ttft_p99 mean = {m:8.1f}ms  [{flag}]")
