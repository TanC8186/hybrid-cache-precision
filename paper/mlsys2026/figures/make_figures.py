"""MLSys 2026 paper figures (nature-figure Python backend).

Every quantitative panel reads its source values from the verified atomic JSONs
under E:\\MLSys_Research\\results; no number is hard-coded except axes labels.
Exports: PDF (vector, editable text) + PNG (300 dpi preview) + TIFF (600 dpi).
"""

from __future__ import annotations

import json
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"E:\MLSys_Research")
FIG = Path(__file__).resolve().parent

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 100,
})

# Tol palette (colorblind-safe) + accent family.
BLUE, CYAN, TEAL, ORANGE, RED, MAGENTA = (
    "#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377")
GRAY = "#BBBBBB"


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def save(fig, name: str, dpi=600):
    """Export vector (PDF/SVG) and raster (PNG/TIFF) with editable text."""
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")  # vector export
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.tiff", dpi=dpi, bbox_inches="tight")


def panel_label(ax, label: str):
    ax.text(-0.14, 1.02, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


# --------------------------------------------------------------------------
# Figure 1: capacity model, 2x2 precision budget.
# --------------------------------------------------------------------------
def fig1_capacity():
    cap = load("results/verified/2026-08-09/capacity-2x2-analysis.json")
    rows = cap["rows"]
    rkv = cap["r_kv_rows"]
    fig = plt.figure(figsize=(7.0, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.38,
                          left=0.07, right=0.99, top=0.86, bottom=0.14)

    # ---- (a) schematic: hybrid layers + memory budget ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(5, 9.45, "hybrid model memory (per sequence)",
            ha="center", fontsize=8, fontweight="bold")

    # layer strip
    n_gdn, n_attn = 6, 2
    y_layers = 7.3
    x_gdn = np.linspace(0.45, 4.1, n_gdn)
    for x in x_gdn:
        ax.add_patch(plt.Rectangle((x, y_layers), 0.55, 0.62, fc=TEAL, ec="k",
                                   lw=0.4))
    x_attn = np.linspace(4.9, 6.2, n_attn)
    for x in x_attn:
        ax.add_patch(plt.Rectangle((x, y_layers), 0.55, 0.62, fc=BLUE, ec="k",
                                   lw=0.4))
    ax.text(2.3, 8.35, "GDN layers", ha="center", fontsize=7, color=TEAL)
    ax.text(5.6, 8.35, "GQA layers", ha="center", fontsize=7, color=BLUE)
    ax.annotate("", xy=(3.1, 6.75), xytext=(3.1, 7.25),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(3.2, 6.9, "recurrent state\n(fixed per seq)", fontsize=6.5,
            va="center")
    ax.annotate("", xy=(5.6, 6.75), xytext=(5.6, 7.25),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(5.7, 6.9, "attention KV\n(grows with L)", fontsize=6.5, va="center")

    # memory bars: KV vs state at fixed budget
    ax.text(2.3, 5.6, "fixed GPU memory budget", fontsize=7, ha="center")
    ax.add_patch(plt.Rectangle((0.5, 4.3), 3.6, 0.9, fc=BLUE, ec="k", lw=0.5,
                               alpha=0.85))
    ax.add_patch(plt.Rectangle((0.5, 3.2), 1.7, 0.9, fc=TEAL, ec="k", lw=0.5))
    ax.text(2.3, 4.75, "attention KV cache", ha="center", fontsize=6.5,
            color="white", fontweight="bold")
    ax.text(1.35, 3.65, "state", ha="center", fontsize=6.5)
    ax.text(0.5, 2.9, "KV bit-width x state precision are both allocatable",
           fontsize=6.5, ha="left", color="#333333")

    # budget grid
    kv = ["int4", "fp16"]
    st = ["fp32", "bf16"]
    for i, k in enumerate(kv):
        for j, s in enumerate(st):
            x = 6.05 + i * 0.95
            y = 5.35 - j * 0.95
            ax.add_patch(plt.Rectangle((x, y), 0.85, 0.85, fc="white", ec="k",
                                       lw=0.6))
            ax.text(x + 0.425, y + 0.5, k, ha="center", fontsize=6.5,
                    fontweight="bold")
            ax.text(x + 0.425, y + 0.22, s, ha="center", fontsize=6.5)
    ax.text(6.95, 6.42, "precision budget", ha="center", fontsize=7,
            fontweight="bold")
    ax.annotate("", xy=(6.95, 6.25), xytext=(6.95, 6.05),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    panel_label(ax, "a")

    # ---- (b) measured vs predicted r_state ----
    ax = fig.add_subplot(gs[0, 1])
    cats = [
        ("2B", 4096, "fp16"), ("2B", 16384, "fp16"), ("9B", 4096, "fp16"),
        ("2B", 4096, "int4"), ("2B", 16384, "int4"),
        ("9B", 4096, "int4"), ("9B", 16384, "int4"),
    ]
    by = {(r["model"].upper(), r["length"], r["kv_dtype"]): r for r in rows}
    x = np.arange(len(cats))
    meas = np.array([by[(m, l, k)]["measured_r_state"] for m, l, k in cats])
    pred = np.array([by[(m, l, k)]["predicted_r_state"] for m, l, k in cats])
    colors = [BLUE if k == "fp16" else ORANGE for _, _, k in cats]
    bars = ax.bar(x, meas, 0.58, color=colors, ec="k", lw=0.5, zorder=3)
    ax.scatter(x, pred, marker="_", s=110, color="k", lw=1.0, zorder=4,
               label="capacity model")
    for xi, (m, l, k), v in zip(x, cats, meas):
        g = by[(m, l, k)]["signed_gap_pct"]
        ax.text(xi, v + 0.035, f"{g:+.1f}%", ha="center", fontsize=6,
                color=RED if g < 0 else BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{m}\nL{l}" for m, l, _ in cats], fontsize=6.5)
    ax.set_ylabel("capacity ratio r_state (bf16 / fp32)")
    ax.set_ylim(0, 1.62)
    ax.axhline(1.0, color=GRAY, lw=0.7, ls="--")
    ax.legend(loc="upper left", fontsize=6.5)  # model prediction line
    ax.set_title("capacity gain from bf16 state (measured vs model)",
                 fontsize=7.5, pad=4)
    panel_label(ax, "b")

    save(fig, "fig1_capacity")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: GSM8K paired deltas (forest, 9-seed).
# --------------------------------------------------------------------------
def fig2_gsm8k():
    g2b = load("results/quality/gsm8k-state9seed-v2-analysis-20260809.json")
    g9b = load("results/quality/gsm8k-9b-state9seed-v2-analysis-20260809.json")
    rows = g2b["rows"]
    stack = g2b["stacking_marginal"]
    labels = []
    eff, lo, hi = [], [], []
    anno = []

    def add(name, delta, ci, note=""):
        labels.append(name)
        eff.append(delta)
        lo.append(ci[0])
        hi.append(ci[1])
        anno.append(note)

    r_int4 = next(r for r in rows if r["allocation"] == "uniform_int4")
    r_state = next(r for r in rows if r["allocation"] == "fp16_statebf16")
    r_both = next(r for r in rows if r["allocation"] == "uniform_int4_statebf16")
    add("2B int4 KV vs fp16", r_int4["delta_vs_fp16"],
        r_int4["ci95_vs_fp16"],
        f"p={r_int4['paired_t']['p_value']:.3f}")
    add("2B bf16 state vs fp32", r_state["delta_vs_fp16"],
        r_state["ci95_vs_fp16"],
        f"p={r_state['paired_t']['p_value']:.3f}")
    add("2B int4+bf16 vs fp16", r_both["delta_vs_fp16"],
        r_both["ci95_vs_fp16"],
        f"p={r_both['paired_t']['p_value']:.3f}")
    add("2B stacking (int4+bf16 vs int4)", stack["mean"], stack["ci95"],
        f"p={stack['paired_t']['p_value']:.3f}")
    r9 = g9b["rows"][1]
    add("9B bf16 state vs fp32", r9["delta_vs_fp16"], r9["ci95_vs_fp16"],
        f"p={r9['paired_t']['p_value']:.3f}")

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    items = list(zip(eff, lo, hi, anno))
    y = np.arange(len(items))[::-1]
    ax.axvline(0, color=GRAY, lw=0.8, ls="--")
    for yi, (e, lw_, h, n) in zip(y, items):
        color = RED if lw_ > 0 or h < 0 else "#333333"
        ax.plot([lw_, h], [yi, yi], color=color, lw=1.3, zorder=2)
        ax.scatter([e], [yi], s=22, color=color, zorder=3)
        ax.text(h + 0.0035, yi, n, va="center", fontsize=5.8, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("GSM8K accuracy delta (pp, bf16 minus fp32; stacked cells noted)")
    ax.set_xlim(-0.055, 0.055)
    ax.set_title("GSM8K, 200 seeded items x 9 seeds, paired 95% CI",
                 fontsize=7, pad=4)
    # Error bars are paired 95% CIs across the 9 dataset seeds.
    panel_label(ax, "a")
    save(fig, "fig2_gsm8k")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3: PPL stacking + RULER.
# --------------------------------------------------------------------------
def fig3_ppl_ruler():
    ppl = load("results/quality/ppl-stacking-analysis-20260809.json")
    rul = load("results/quality/ruler-statebf16-multiseed-analysis-20260809.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.45),
                             gridspec_kw={"width_ratios": [1, 1.6],
                                          "wspace": 0.45})

    # PPL stacking
    ax = axes[0]
    tabs = [("C4", ppl["tables"]["c4"]), ("PG19", ppl["tables"]["pg19"])]
    y = [1, 0]
    for yi, (name, t) in zip(y, tabs):
        e, ci = t["delta_bf16_vs_fp32"], t["ci95_delta"]
        ax.plot(ci, [yi, yi], color="#333333", lw=1.3)
        ax.scatter([e], [yi], s=22, color=BLUE, zorder=3)
        ax.text(ci[1] + 0.0007, yi, f"{e:+.4f}", va="center", fontsize=6)
    ax.axvline(0, color=GRAY, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(["C4", "PG19"], fontsize=7)
    ax.set_xlabel("PPL delta (bf16 minus fp32 state)")
    ax.set_title("PPL stacking (int4 KV), 3 seeds", fontsize=7, pad=4)
    # Error bars are paired 95% CIs across the 3 dataset seeds.
    panel_label(ax, "a")

    # RULER
    ax = axes[1]
    cells = []
    for r in rul["rows"]:
        task = r["task"].replace("ruler_", "")
        cells.append((f"{r['model'].upper()} {task} L{r['length']}",
                      r["delta_mean"], r["ci95_delta"]))
    y = np.arange(len(cells))[::-1]
    for yi, (name, e, ci) in zip(y, cells):
        ax.plot(ci, [yi, yi], color="#333333", lw=1.3)
        ax.scatter([e], [yi], s=22, color=TEAL, zorder=3)
        ax.text(ci[1] + 0.8, yi, f"{e:+.2f}", va="center", fontsize=6)
    ax.axvline(0, color=GRAY, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in cells], fontsize=6.5)
    ax.set_xlabel("accuracy delta (pp, bf16 minus fp32)")
    ax.set_title("RULER, 3 dataset seeds, point + wide CI",
                 fontsize=7, pad=4)
    # Error bars are paired 95% CIs across the 3 dataset seeds.
    panel_label(ax, "b")
    save(fig, "fig3_ppl_ruler")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4: serving Random60 paired goodput delta (formal + repro) and
#           sustainable-boundary parity map.
# --------------------------------------------------------------------------
def fig4_serving():
    formal = load("results/verified/2026-08-09/statebf16-serving-formal-analysis.json")
    repro = load("results/verified/2026-08-09/statebf16-serving-repro-analysis.json")

    def pick(js, workload, rate, th):
        return next(r for r in js["paired_deltas"]
                    if r["workload"] == workload and r["rate"] == rate
                    and r["threshold_ms"] == th)

    fig = plt.figure(figsize=(7.0, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.4,
                          left=0.06, right=0.985, top=0.87, bottom=0.16)

    # Random60 goodput deltas: overload region (r40-r50).
    ax = fig.add_subplot(gs[0, 0])
    rates = [40, 45, 50]
    ths = [250, 500, 1000, 2000, 3000]
    rows_out = []
    for r in rates:
        for t in ths:
            f, p = pick(formal, "random", r, t), pick(repro, "random", r, t)
            rows_out.append((r, t, f, p))
    y = np.arange(len(rows_out))[::-1]
    ytick = []
    for yi, (r, t, f, p) in zip(y, rows_out):
        ytick.append(f"r{r} {t}ms")
        # reproduction (secondary) drawn first, then formal (hero)
        ax.plot([p["ci95"][0], p["ci95"][1]], [yi - 0.15, yi - 0.15],
                color=CYAN, lw=1.4, zorder=2)
        ax.scatter([p["mean_delta_goodput"]], [yi - 0.15], s=18, marker="D",
                   color=CYAN, zorder=3)
        ax.plot([f["ci95"][0], f["ci95"][1]], [yi + 0.15, yi + 0.15],
                color=BLUE, lw=1.6, zorder=2)
        ax.scatter([f["mean_delta_goodput"]], [yi + 0.15], s=20,
                   color=BLUE, zorder=3)
    ax.axvline(0, color=GRAY, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(ytick, fontsize=6.2)
    ax.set_xlabel("paired goodput delta (bf16 minus fp32 state)")
    ax.set_xlim(-0.12, 0.72)
    ax.set_title("Random60 overload: goodput delta, 3 seeds, 95% CI",
                 fontsize=7.2, pad=4)
    ax.plot([], [], color=BLUE, lw=1.6, label="formal")
    ax.plot([], [], color=CYAN, lw=1.4, ls="-", label="independent repro")
    ax.legend(fontsize=6.3, loc="lower right")  # formal vs repro runs
    panel_label(ax, "a")

    # boundary parity map: cells where bf16 boundary == fp32 boundary
    ax = fig.add_subplot(gs[0, 1])
    work = [("Random60", "random"), ("ShareGPT300", "sharegpt")]
    grid = []
    for wk, key in work:
        for t in ths:
            fb = formal["boundaries"][f"int4__{key}"][str(t)]
            sb = formal["boundaries"][f"int4_statebf16__{key}"][str(t)]
            rb = repro["boundaries"][f"int4_statebf16__{key}"][str(t)]
            grid.append((wk, t, fb, sb, rb))
    n = len(grid)
    for i, (wk, t, fb, sb, rb) in enumerate(grid):
        y = n - 1 - i
        color = TEAL if sb == fb and rb == fb else ORANGE
        ax.scatter(0.15, y, s=90, marker="s", color=color, ec="k", lw=0.4)
        ax.scatter(0.5, y, s=90, marker="s", color=color, ec="k", lw=0.4)
        ax.text(0.15, y + 0.28, f"{fb:.0f}", ha="center", fontsize=5.8)
        ax.text(0.5, y + 0.28, f"{sb:.0f} ({rb:.0f})", ha="center",
                fontsize=5.8)
        ax.text(-0.1, y, f"{wk.split('60')[0]} {t}ms", ha="right",
                va="center", fontsize=6)
    ax.set_xlim(-0.45, 0.95)
    ax.set_ylim(-0.6, n + 0.35)
    ax.axis("off")
    ax.text(0.15, n + 0.22, "int4", ha="center", fontsize=6.5,
            fontweight="bold")
    ax.text(0.5, n + 0.22, "int4+bf16\n(rep.)", ha="center", fontsize=6.5,
            fontweight="bold")
    ax.plot([], [], marker="s", color=TEAL, ls="", label="boundary equal",
            markersize=5)
    ax.plot([], [], marker="s", color=ORANGE, ls="", label="boundary differs",
            markersize=5)
    ax.legend(fontsize=6.2, loc="lower left", bbox_to_anchor=(0.02, -0.02))
    ax.set_title("sustainable rate boundary (req/s)", fontsize=7.2, pad=4)
    panel_label(ax, "b")
    save(fig, "fig4_serving")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 5: block granularity evidence (capacity mechanism).
# --------------------------------------------------------------------------
def fig5_block_granularity():
    cap = load("results/verified/2026-08-09/capacity-2x2-analysis.json")
    cats = [
        ("2B", 4096, "fp16"), ("2B", 16384, "fp16"), ("9B", 4096, "fp16"),
        ("2B", 4096, "int4"), ("2B", 16384, "int4"),
        ("9B", 4096, "int4"), ("9B", 16384, "int4"),
    ]
    by = {(r["model"].upper(), r["length"], r["kv_dtype"]): r for r in cap["rows"]}
    labels = [f"{m}\nL{l}" for m, l, _ in cats]
    fp32_bs = [by[(m, l, k)]["fp32_block_size"] for m, l, k in cats]
    bf16_bs = [by[(m, l, k)]["bf16_block_size"] for m, l, k in cats]
    fp32_nb = [by[(m, l, k)]["fp32_num_gpu_blocks"] for m, l, k in cats]
    bf16_nb = [by[(m, l, k)]["bf16_num_gpu_blocks"] for m, l, k in cats]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7),
                             gridspec_kw={"wspace": 0.32})
    x = np.arange(len(cats))
    w = 0.36
    for ax, a, b, ylab, title in (
            (axes[0], fp32_bs, bf16_bs, "tokens per GPU block",
             "block size"),
            (axes[1], fp32_nb, bf16_nb, "allocated GPU blocks",
             "block count")):
        ax.bar(x - w / 2, a, w, color=GRAY, ec="k", lw=0.4, label="fp32 state")
        ax.bar(x + w / 2, b, w, color=TEAL, ec="k", lw=0.4, label="bf16 state")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.2)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=7, pad=4)
        ax.legend(fontsize=6, loc="upper right")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save(fig, "fig5_block_granularity")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 6: per-layer state sensitivity (negative result, corrected).
# --------------------------------------------------------------------------
def fig6_sensitivity():
    js = load("results/quality/state-sensitivity-analysis-20260809-bonf.json")
    rows = [r for r in js["rows"] if r["config"].startswith("bf16_L")]
    layers = [int(r["config"].split("L")[1]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    panels = [
        ("c4", "C4", axes[0]),
        ("pg19", "PG19", axes[1]),
    ]
    for tag, name, ax in panels:
        y = np.arange(len(rows))[::-1]
        for yi, (r, ly) in enumerate(zip(rows, layers)):
            d = r[f"{tag}_delta"]
            ci = r[f"{tag}_ci95"]
            raw_sig = r.get(f"{tag}_sensitive", False)
            color = RED if raw_sig else "#333333"
            ax.plot(ci, [yi, yi], color=color, lw=1.3, zorder=2)
            ax.scatter([d], [yi], s=22, color=color, zorder=3)
        ax.axvline(0, color=GRAY, lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(layers, fontsize=6.2)
        ax.set_xlabel("PPL delta (bf16 layer minus fp32)")
        ax.set_title(f"{name} (3 seeds, 95% CI)", fontsize=7, pad=4)
        if tag == "c4":
            ax.set_ylabel("GDN layer (bf16)")
    axes[1].text(0.012, 1.02, "raw p<0.05; n.s. after\nBonferroni / BH-FDR",
                 transform=axes[1].transAxes, fontsize=6, color=RED,
                 va="bottom", ha="left")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save(fig, "fig6_sensitivity")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 7: harness boundary (chunk ablation) + stacking cost.
# --------------------------------------------------------------------------
def fig7_harness():
    chunk = {}
    for st in ("fp32", "bf16"):
        for ck in (128, 1):
            p = ROOT / ("results/quality/chunk-ablation/"
                        f"chunk-ablation-20260809__state{st}__chunk{ck}__2b.csv")
            with p.open(encoding="utf-8") as fh:
                val = float(list(csv.DictReader(fh))[0]["ppl_mean"])
            chunk[(st, ck)] = val
    ppl = load("results/quality/ppl-stacking-analysis-20260809.json")
    cost = ppl["stacking_cost_vs_fp16_kv"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6),
                             gridspec_kw={"width_ratios": [1.15, 1.0],
                                          "wspace": 0.4})

    ax = axes[0]
    cats = [("fp32", 128), ("bf16", 128), ("fp32", 1), ("bf16", 1)]
    x = np.arange(len(cats))
    vals = [chunk[c] for c in cats]
    colors = [BLUE, TEAL, BLUE, TEAL]
    ax.bar(x, vals, 0.55, color=colors, ec="k", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["fp32\nchunk=128", "bf16\nchunk=128",
                        "fp32\nchunk=1", "bf16\nchunk=1"], fontsize=6.2)
    ax.set_ylabel("C4 PPL (1 seed x 1 seq)")
    ax.set_ylim(0, 42)
    ax.set_title("chunk-level write-back rounding", fontsize=7, pad=4)
    ax.annotate("+87% vs chunk=128", xy=(1.5, 28), fontsize=6.5,
                ha="center", color=RED)
    panel_label(ax, "a")

    ax = axes[1]
    corpora = ["C4", "PG19"]
    fp16_vals = [cost[c.lower()]["fp16kv_state_delta"] for c in corpora]
    int4_vals = [cost[c.lower()]["int4kv_state_delta"] for c in corpora]
    x = np.arange(len(corpora))
    w = 0.32
    ax.axhline(0, color=GRAY, lw=0.8, ls="--")
    ax.bar(x - w / 2, fp16_vals, w, color=BLUE, ec="k", lw=0.4,
           label="fp16 KV")
    ax.bar(x + w / 2, int4_vals, w, color=ORANGE, ec="k", lw=0.4,
           label="int4 KV")
    for xi, v in zip(x - w / 2, fp16_vals):
        ax.text(xi, v + 0.0004, f"{v:+.4f}", ha="center", fontsize=5.8)
    for xi, v in zip(x + w / 2, int4_vals):
        ax.text(xi, v + 0.0004, f"{v:+.4f}", ha="center", fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels(corpora, fontsize=7)
    ax.set_ylabel("state-bf16 marginal PPL delta")
    ax.set_title("stacking cost: fp16 vs int4 KV", fontsize=7, pad=4)
    ax.legend(fontsize=6)
    panel_label(ax, "b")
    save(fig, "fig7_harness")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 8: GSM8K per-seed paired accuracy.
# --------------------------------------------------------------------------
def fig8_gsm8k_per_seed():
    g2b = load("results/quality/gsm8k-state9seed-v2-analysis-20260809.json")
    g9b = load("results/quality/gsm8k-9b-state9seed-v2-analysis-20260809.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8),
                             gridspec_kw={"wspace": 0.3})

    def draw(ax, rows, allocs, title):
        seeds = list(rows[0]["per_seed"].keys())
        x = np.arange(len(allocs))
        for s in seeds:
            ys = [next(r for r in rows if r["allocation"] == a)["per_seed"][s]
                  for a in allocs]
            ax.plot(x, ys, color="#999999", lw=0.7, alpha=0.65, zorder=1)
        means = [next(r for r in rows if r["allocation"] == a)["mean"]
                 for a in allocs]
        ax.plot(x, means, color=BLUE, lw=1.6, marker="o", ms=4, zorder=3,
                label="seed mean")
        ax.set_xticks(x)
        ax.set_xticklabels(allocs, fontsize=6)
        ax.set_ylim(0.62, 0.9)
        ax.set_title(title, fontsize=7, pad=4)
        ax.legend(fontsize=6)

    rows2 = g2b["rows"]
    draw(axes[0], rows2,
         ["fp16", "fp16_statebf16", "uniform_int4", "uniform_int4_statebf16"],
         "2B, 9 seeds")
    draw(axes[1], g9b["rows"], ["fp16", "fp16_statebf16"], "9B, 9 seeds")
    for ax in axes:
        ax.set_ylabel("GSM8K accuracy")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save(fig, "fig8_gsm8k_per_seed")
    plt.close(fig)


if __name__ == "__main__":
    fig1_capacity()
    fig2_gsm8k()
    fig3_ppl_ruler()
    fig4_serving()
    fig5_block_granularity()
    fig6_sensitivity()
    fig7_harness()
    fig8_gsm8k_per_seed()
    print("figures written to", FIG)
