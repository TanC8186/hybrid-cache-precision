import json, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---- MANDATORY publication style ----
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.fontsize'] = 8.5

# restrained palette
FP16_C = '#4D4D4D'    # neutral dark (baseline)
INT4_C = '#0F4D92'    # blue main (method)
SLO_C  = '#B64342'    # red strong (SLO line)

BASE = "E:/MLSys_Research/results/ablations/bench_lat"
OUT  = "E:/MLSys_Research/results/figures"
os.makedirs(OUT, exist_ok=True)

def load(variant):
    rows = []
    for fp in sorted(glob.glob(f"{BASE}/{variant}/openai-*.json")):
        d = json.load(open(fp))
        rows.append(dict(
            rate=d["request_rate"], req_s=d["request_throughput"],
            out_tok_s=d["output_throughput"],
            ttft_p99=d["p99_ttft_ms"], ttft_mean=d["mean_ttft_ms"],
            tpot_p50=d["median_tpot_ms"], tpot_p99=d["p99_tpot_ms"],
        ))
    return sorted(rows, key=lambda r: r["rate"])

int4 = load("int4")
fp16 = load("fp16")

# ============ Fig 1: latency-throughput (TTFT p99 vs goodput req/s) ============
fig, ax = plt.subplots(figsize=(4.2, 3.3))
ax.plot([r["req_s"] for r in fp16], [r["ttft_p99"] for r in fp16],
        color=FP16_C, lw=1.6, marker='o', ms=4, label="fp16 baseline")
ax.plot([r["req_s"] for r in int4], [r["ttft_p99"] for r in int4],
        color=INT4_C, lw=1.6, marker='s', ms=4, label="uniform int4")
ax.set_xlabel("Goodput (req/s)")
ax.set_ylabel("TTFT p99 (ms)")
ax.set_title("TTFT p99 vs served throughput", fontsize=10)
ax.legend(loc="upper left")
ax.set_xlim(0, 42)
ax.set_ylim(0, 4500)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_latency_throughput.png", dpi=150)
plt.close(fig)

# ============ Fig 2: SLO TTFT p99 vs offered rate ============
fig, ax = plt.subplots(figsize=(4.6, 3.3))
ax.plot([r["rate"] for r in fp16], [r["ttft_p99"] for r in fp16],
        color=FP16_C, lw=1.6, marker='o', ms=4, label="fp16 baseline")
ax.plot([r["rate"] for r in int4], [r["ttft_p99"] for r in int4],
        color=INT4_C, lw=1.6, marker='s', ms=4, label="uniform int4")
ax.axhline(2000, color=SLO_C, lw=1.2, ls='--', zorder=0)
ax.text(0.5, 2060, "SLO: TTFT p99 < 2000 ms", color=SLO_C, fontsize=8)
# annotate max SLO-satisfying rates
ax.annotate("max SLO rate = 50 req/s", xy=(50, 1574), xytext=(28, 3300),
            color=INT4_C, fontsize=8, arrowprops=dict(arrowstyle='->', color=INT4_C, lw=0.9))
ax.annotate("max SLO rate = 40 req/s", xy=(40, 566), xytext=(8, 3300),
            color=FP16_C, fontsize=8, arrowprops=dict(arrowstyle='->', color=FP16_C, lw=0.9))
ax.set_xlabel("Offered load (req/s)")
ax.set_ylabel("TTFT p99 (ms)")
ax.set_title("SLO check: TTFT p99 vs offered rate", fontsize=10)
ax.legend(loc="upper left")
ax.set_xlim(0, 80)
ax.set_ylim(0, 4800)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_slo_ttft.png", dpi=150)
plt.close(fig)

# ============ Fig 3: TPOT p50 / p99 vs rate ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0), sharex=True)
for ax, col, title in [(ax1, "p50", "TPOT p50"), (ax2, "p99", "TPOT p99")]:
    ax.plot([r["rate"] for r in fp16], [r["tpot_p50"] if col=="p50" else r["tpot_p99"] for r in fp16],
            color=FP16_C, lw=1.6, marker='o', ms=3.5, ls='-' if col=="p50" else '--',
            label="fp16 baseline" if col=="p50" else None)
    ax.plot([r["rate"] for r in int4], [r["tpot_p50"] if col=="p50" else r["tpot_p99"] for r in int4],
            color=INT4_C, lw=1.6, marker='s', ms=3.5, ls='-' if col=="p50" else '--',
            label="uniform int4" if col=="p50" else None)
    ax.set_xlabel("Offered load (req/s)")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 60)
ax1.legend(loc="upper left")
fig.suptitle("TPOT: per-step latency vs offered rate", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_tpot.png", dpi=150)
plt.close(fig)

print("saved to", OUT)
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        print(" ", f, os.path.getsize(os.path.join(OUT, f)), "bytes")
