import json, glob, os, math

BASE = "E:/MLSys_Research/results/ablations/bench_lat"
rows = []
for variant in ["int4", "fp16"]:
    for fp in sorted(glob.glob(f"{BASE}/{variant}/openai-*.json")):
        d = json.load(open(fp))
        rate = d["request_rate"]
        rows.append(dict(
            variant=variant,
            rate=rate,
            req_s=d["request_throughput"],
            out_tok_s=d["output_throughput"],
            tot_tok_s=d["total_token_throughput"],
            ttft_mean=d["mean_ttft_ms"],
            ttft_p50=d["median_ttft_ms"],
            ttft_p99=d["p99_ttft_ms"],
            tpot_mean=d["mean_tpot_ms"],
            tpot_p50=d["median_tpot_ms"],
            tpot_p99=d["p99_tpot_ms"],
            peak_conc=d["max_concurrent_requests"],
            completed=d["completed"],
            failed=d["failed"],
            duration=d["duration"],
            total_out_tokens=d["total_output_tokens"],
            max_out_tok_s=d["max_output_tokens_per_s"],
        ))

def fmt(x): return f"{x:.4f}"
def show(variant):
    print(f"===== {variant} =====")
    print(f"{'rate':>5} {'req_s':>9} {'out_tok/s':>11} {'TTFTmean':>9} {'TTFTp50':>9} {'TTFTp99':>9} {'TPOTmean':>9} {'TPOTp50':>9} {'TPOTp99':>9} {'peakConc':>8} {'fail':>4} {'dur':>7}")
    for r in sorted([x for x in rows if x["variant"]==variant], key=lambda x:x["rate"]):
        print(f"{r['rate']:>5.0f} {r['req_s']:>9.4f} {r['out_tok_s']:>11.2f} {r['ttft_mean']:>9.2f} {r['ttft_p50']:>9.2f} {r['ttft_p99']:>9.2f} {r['tpot_mean']:>9.3f} {r['tpot_p50']:>9.3f} {r['tpot_p99']:>9.3f} {r['peak_conc']:>8.0f} {r['failed']:>4} {r['duration']:>7.1f}")

for v in ["int4","fp16"]:
    show(v)

# SLO analysis
SLO_TTFT = 2000
SLO_TPOT = 200
print("\n===== SLO check (TTFT p99 < 2000ms AND TPOT p99 < 200ms) =====")
for v in ["int4","fp16"]:
    print(f"--- {v} ---")
    max_slo_rate = None
    for r in sorted([x for x in rows if x["variant"]==v], key=lambda x:x["rate"]):
        ok = r["ttft_p99"] < SLO_TTFT and r["tpot_p99"] < SLO_TPOT
        mark = "OK " if ok else "FAIL"
        if ok: max_slo_rate = r["rate"]
        print(f"  R={r['rate']:>5.0f}: TTFTp99={r['ttft_p99']:>9.1f}ms TPOTp99={r['tpot_p99']:>7.1f}ms -> {mark}")
    print(f"  MAX SLO-satisfying rate: {max_slo_rate}")

# Saturation / plateau (goodput)
print("\n===== Saturation analysis (goodput req/s plateau) =====")
for v in ["int4","fp16"]:
    d = sorted([x for x in rows if x["variant"]==v], key=lambda x:x["rate"])
    print(f"--- {v} ---")
    for r in d:
        print(f"  offered={r['rate']:>5.0f}  goodput={r['req_s']:>8.4f}  out_tok/s={r['out_tok_s']:>9.2f}  peak_conc={r['peak_conc']:>5.0f}")

# capacity ratio
print("\n===== Capacity ratio =====")
print("int4 KV tokens:", 2701721, "fp16 KV tokens:", 1203106, "ratio:", 2701721/1203106)

# low-load TTFT comparison R=1
print("\n===== R=1 TTFT p99 =====")
for v in ["int4","fp16"]:
    r = [x for x in rows if x["variant"]==v and x["rate"]==1][0]
    print(f"  {v}: TTFT p99={r['ttft_p99']:.1f}ms  TTFT p50={r['ttft_p50']:.1f}ms  TPOT p50={r['tpot_p50']:.2f}ms  TPOT p99={r['tpot_p99']:.2f}ms")
