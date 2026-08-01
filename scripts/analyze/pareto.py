"""分析字节预算排序实验结果：PPL vs KV 字节的 Pareto 前沿 + 等字节排序对比。

输入: results/ablations/byte_budget_ordering.csv
      (bits, evict_budget, ppl, kv_quant_bytes, kv_fp16_bytes, time_s)
输出:
  1. 控制台打印：每个字节预算下最优方法（排序结论）
  2. results/figures/byte_budget_pareto.png：PPL vs KV 字节散点 + Pareto 前沿

核心问题：在给定 KV 字节预算下，"驱逐部分 token 用更高位宽" vs "全保留用更低位宽"，
哪个 PPL 更好？
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return [dict(r) for r in csv.DictReader(f)]


def analyze(rows: list[dict], seq_len: int) -> None:
    # 每条：估算保留 token 数（无驱逐 = seq_len，驱逐 = evict_budget）
    pts = []
    for r in rows:
        bits = int(float(r["bits"]))
        evict = int(float(r["evict_budget"] or 0))
        ppl = float(r["ppl"])
        qbytes = float(r["kv_quant_bytes"])
        keep = evict if evict > 0 else seq_len
        bpt = qbytes / keep  # bytes per kept token
        pts.append({"bits": bits, "keep": keep, "ppl": ppl, "bytes": qbytes, "bpt": bpt})

    print(f"{'bits':>4} {'keep':>5} {'bytes':>9} {'PPL':>8}  备注")
    print("-" * 50)
    for p in sorted(pts, key=lambda x: (x["bytes"], x["ppl"])):
        note = ""
        if p["bytes"] == max(x["bytes"] for x in pts):
            note = "← FP16 baseline"
        print(f"{p['bits']:>4} {p['keep']:>5} {p['bytes']:>9.0f} {p['ppl']:>8.3f}  {note}")

    # 等字节排序：把字节分成区间，找每区间最优 PPL 的方法
    print("\n=== 等字节区间最优（排序结论） ===")
    budget_candidates = sorted({p["bytes"] for p in pts})
    for target in budget_candidates:
        # 找与 target 字节最接近（±3%）的其他方法
        peers = [p for p in pts if abs(p["bytes"] - target) / target < 0.04]
        if len(peers) < 2:
            continue
        best = min(peers, key=lambda p: p["ppl"])
        print(f"\n字节预算 ~{target:,.0f}: 最优 {best['bits']}bit/keep{best['keep']} (PPL {best['ppl']:.3f})")
        for p in sorted(peers, key=lambda x: x["ppl"]):
            tag = " ← 最优" if p is best else ""
            print(f"   {p['bits']}bit/keep{p['keep']}  bytes={p['bytes']:,.0f}  PPL={p['ppl']:.3f}{tag}")


def plot(rows: list[dict], seq_len: int, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = [int(float(r["evict_budget"] or 0)) or seq_len for r in rows]
    bits = [int(float(r["bits"])) for r in rows]
    ppl = [float(r["ppl"]) for r in rows]
    bytes_ = [float(r["kv_quant_bytes"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {2: "tab:red", 3: "tab:orange", 4: "tab:green", 8: "tab:blue", 16: "black"}
    for b in sorted(set(bits)):
        idx = [i for i, x in enumerate(bits) if x == b]
        ax.scatter([bytes_[i] for i in idx], [ppl[i] for i in idx],
                   label=f"{b}-bit", color=colors.get(b), s=50)
        for i in idx:
            ax.annotate(f"{keep[i]}", (bytes_[i], ppl[i]), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("KV bytes (log)")
    ax.set_ylabel("PPL (lower better)")
    ax.set_title(f"KV cache byte budget ordering — Qwen3.5-2B (6 GQA layers, seq_len={seq_len})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"→ {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/ablations/byte_budget_ordering.csv")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--fig", default="results/figures/byte_budget_pareto.png")
    args = ap.parse_args()

    rows = load_csv(Path(args.csv))
    analyze(rows, args.seq_len)
    plot(rows, args.seq_len, Path(args.fig))


if __name__ == "__main__":
    main()
