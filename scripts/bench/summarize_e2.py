#!/usr/bin/env python3
"""One-line summary per vllm bench result JSON in a directory.
Usage: summarize_e2.py <dir> [label]
Columns: rate, req_throughput, output_tok/s, ttft_p50, ttft_p99, tpot_p50, tpot_p99, itl_p99, completed, failed
"""
import json, glob, os, sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(os.path.normpath(d))

files = sorted(glob.glob(os.path.join(d, "*.json")))
if not files:
    print(f"NO_JSON {d}")
    sys.exit(0)

print(f"## {label}  ({len(files)} files)")
print("rate   req_tp   out_tok/s  ttft_p50  ttft_p99  tpot_p50  tpot_p99  itl_p99  completed failed")
for f in files:
    j = json.load(open(f))
    print(f"{j['request_rate']:>5.0f} {j['request_throughput']:7.2f} {j['output_throughput']:10.1f} "
          f"{j['median_ttft_ms']:8.1f} {j['p99_ttft_ms']:8.1f} {j['median_tpot_ms']:8.2f} "
          f"{j['p99_tpot_ms']:8.2f} {j['p99_itl_ms']:7.1f} {j['completed']:9d} {j['failed']:d}")
