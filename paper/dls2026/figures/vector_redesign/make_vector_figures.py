"""Vector-first MLSys figures for the hybrid precision-budget paper.

The script reads the repository's verified/quality artifacts and exports only
editable-text SVG and TrueType-text PDF files into this directory.  Raster PNG
files are written only to tmp/figure-redesign/previews for visual QA.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
PREVIEW = ROOT / "tmp" / "figure-redesign" / "previews"

NAVY = "#23364D"
BLUE = "#2A6FBB"
BLUE_LIGHT = "#DCEAF7"
TEAL = "#1B8E8A"
TEAL_LIGHT = "#D9EFED"
ORANGE = "#E58B2A"
ORANGE_LIGHT = "#F9E6CC"
RED = "#C7472D"
GRAY_DARK = "#4E5965"
GRAY = "#9AA3AD"
GRAY_LIGHT = "#E8EBEE"
INK = "#20252B"
WHITE = "#FFFFFF"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.6,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.6,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def save_vector(fig: mpl.figure.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{name}.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PREVIEW / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        PREVIEW / f"{name}.tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.12, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
        clip_on=False,
    )


def clean_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def fmt_signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def draw_ci(
    ax: mpl.axes.Axes,
    y: float,
    estimate: float,
    ci: list[float] | tuple[float, float],
    color: str,
    marker: str = "o",
    size: float = 22,
) -> None:
    ax.plot(ci, [y, y], color=color, lw=1.25, solid_capstyle="round", zorder=2)
    ax.scatter([estimate], [y], color=color, s=size, marker=marker, zorder=3,
               linewidth=0.55, edgecolor=WHITE)


def fig1_capacity() -> None:
    cap = load("results/verified/2026-08-14/capacity-2x2-analysis-corrected.json")
    rows = cap["rows"]
    by = {(r["model"].upper(), r["length"], r["kv_dtype"]): r for r in rows}

    fig = plt.figure(figsize=(7.0, 3.05))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.95, 1.55],
        left=0.035,
        right=0.995,
        top=0.90,
        bottom=0.18,
        wspace=0.27,
    )

    # Panel a: one causal flow, not a collage of independent boxes.
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.95, "Hybrid layer stack", fontsize=7.2, fontweight="bold",
            color=INK, va="top")
    block_y = 0.80
    x0, gap, bw = 0.05, 0.012, 0.067
    for i in range(8):
        is_gqa = i in (3, 7)
        fc = ORANGE if is_gqa else BLUE
        ax.add_patch(
            FancyBboxPatch(
                (x0 + i * (bw + gap), block_y),
                bw,
                0.085,
                boxstyle="round,pad=0.003,rounding_size=0.008",
                fc=fc,
                ec="none",
            )
        )
    ax.text(0.05, 0.765, "many GDN layers", color=BLUE, fontsize=5.9, va="top")
    ax.text(0.63, 0.765, "few GQA layers", color=ORANGE, fontsize=5.9,
            va="top", ha="right")

    # Connectors are drawn before memory nodes so they stay behind labels.
    ax.add_patch(FancyArrowPatch((0.27, 0.78), (0.27, 0.66), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.8, color=GRAY_DARK))
    ax.add_patch(FancyArrowPatch((0.60, 0.78), (0.60, 0.66), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.8, color=GRAY_DARK))

    pool = FancyBboxPatch(
        (0.05, 0.31), 0.90, 0.33,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        fc="#F7F8FA", ec=GRAY, lw=0.8,
    )
    ax.add_patch(pool)
    ax.text(0.08, 0.615, "Fixed GPU memory pool", fontsize=6.5,
            fontweight="bold", color=INK, va="top")

    kv_box = FancyBboxPatch(
        (0.09, 0.45), 0.54, 0.105,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        fc=ORANGE_LIGHT, ec=ORANGE, lw=0.9,
    )
    state_box = FancyBboxPatch(
        (0.67, 0.39), 0.23, 0.165,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        fc=BLUE_LIGHT, ec=BLUE, lw=0.9,
    )
    ax.add_patch(kv_box)
    ax.add_patch(state_box)
    ax.text(0.36, 0.507, "Attention KV", ha="center", va="center",
            fontsize=6.7, fontweight="bold", color=INK)
    ax.text(0.36, 0.469, "per token; grows with L", ha="center", va="center",
            fontsize=5.7, color=GRAY_DARK)
    ax.text(0.785, 0.495, "Recurrent\nstate", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color=INK)
    ax.text(0.785, 0.421, "fixed per sequence", ha="center", va="center",
            fontsize=5.5, color=GRAY_DARK)

    ax.text(0.13, 0.23, "KV dtype", fontsize=6.0, color=ORANGE,
            fontweight="bold", ha="center")
    ax.text(0.36, 0.23, "+", fontsize=8.5, color=GRAY_DARK, ha="center")
    ax.text(0.59, 0.23, "state dtype", fontsize=6.0, color=BLUE,
            fontweight="bold", ha="center")
    ax.text(0.86, 0.23, "joint budget", fontsize=6.3, color=INK,
            fontweight="bold", ha="center")
    ax.add_patch(FancyArrowPatch((0.17, 0.20), (0.79, 0.20), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.9, color=GRAY_DARK))

    ax.text(0.05, 0.08, "Capacity model", fontsize=6.0, color=GRAY_DARK,
            fontweight="bold", va="center")
    ax.text(0.35, 0.08, "N(L) = M / (A L + G);  T(L) = L N(L)", fontsize=6.6,
             color=INK, va="center")
    panel_label(ax, "a", x=-0.02, y=1.01)

    # Panel b: measured/model pairs with the int4 group first.
    ax = fig.add_subplot(gs[0, 1])
    cats = [
        ("2B", 4096, "int4"),
        ("2B", 16384, "int4"),
        ("9B", 4096, "int4"),
        ("9B", 16384, "int4"),
        ("2B", 4096, "fp16"),
        ("2B", 16384, "fp16"),
        ("9B", 4096, "fp16"),
    ]
    x = np.array([0, 1, 2, 3, 4.6, 5.6, 6.6])
    meas = np.array([by[c]["measured_r_state"] for c in cats])
    pred = np.array([by[c]["predicted_r_state"] for c in cats])
    gaps = np.array([by[c]["signed_gap_pct"] for c in cats])
    colors = [ORANGE if c[2] == "int4" else BLUE for c in cats]

    ax.axhline(1.0, color=GRAY, lw=0.8, ls=(0, (3, 2)), zorder=0)
    for xi, m, p, g, color in zip(x, meas, pred, gaps, colors):
        ax.plot([xi, xi], [min(m, p), max(m, p)], color=GRAY_DARK, lw=0.8, zorder=1)
        ax.scatter([xi], [p], s=25, marker="D", facecolor=WHITE,
                   edgecolor=INK, lw=0.8, zorder=2)
        ax.scatter([xi], [m], s=38, marker="o", facecolor=color,
                   edgecolor=WHITE, lw=0.65, zorder=3)
        ax.text(xi, max(m, p) + 0.034, f"{m:.3f}x", ha="center", va="bottom",
                fontsize=5.8, color=color, fontweight="bold")
        ax.text(xi, min(m, p) - 0.035, f"gap {g:+.1f}%", ha="center", va="top",
                fontsize=5.3, color=GRAY_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{l // 1024}K" for m, l, _ in cats])
    ax.set_ylabel("Capacity ratio (bf16 / fp32 state)")
    ax.set_ylim(0.94, 1.52)
    ax.set_xlim(-0.55, 7.05)
    ax.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    ax.text(1.5, 1.505, "int4 KV", color=ORANGE, ha="center", va="bottom",
            fontsize=6.6, fontweight="bold")
    ax.text(5.6, 1.505, "fp16 KV", color=BLUE, ha="center", va="bottom",
            fontsize=6.6, fontweight="bold")
    ax.plot([4.0, 4.0], [0.98, 1.49], color=GRAY_LIGHT, lw=1.0, clip_on=False)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ms=5, color="none", markerfacecolor=GRAY_DARK,
                   markeredgecolor=WHITE, label="measured"),
            Line2D([], [], marker="D", ms=4, color="none", markerfacecolor=WHITE,
                   markeredgecolor=INK, label="capacity model"),
        ],
        loc="lower right",
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    rkv = {(r["model"], r["length"], r["state_dtype"]): r["measured_r_kv"]
           for r in cap["r_kv_rows"]}
    ax.text(
        0.01,
        -0.24,
        "2B - 4K compound KV gain: "
        f"{rkv[('2b', 4096, 'fp32')]:.3f}x  ->  "
        f"{rkv[('2b', 4096, 'bf16')]:.3f}x",
        transform=ax.transAxes,
        fontsize=6.2,
        color=INK,
        fontweight="bold",
        ha="left",
        va="top",
    )
    clean_axis(ax)
    panel_label(ax, "b", x=-0.10, y=1.01)

    save_vector(fig, "fig1_capacity")
    plt.close(fig)


def fig2_gsm8k() -> None:
    g2b = load("results/quality/gsm8k-state9seed-v2-dependence-aware-20260814.json")
    g9b = load("results/quality/gsm8k-9b-state9seed-v2-dependence-aware-20260814.json")
    rows2 = {r["allocation"]: r for r in g2b["rows"]}
    rows9 = {r["allocation"]: r for r in g9b["rows"]}

    entries = [
        ("2B  state bf16 vs fp32", rows2["fp16_statebf16"]),
        ("9B  state bf16 vs fp32", rows9["fp16_statebf16"]),
        ("2B  int4 KV vs fp16", rows2["uniform_int4"]),
        ("2B  int4 + bf16 vs fp16", rows2["uniform_int4_statebf16"]),
    ]
    stack = g2b["stacking_marginal"]
    entries.append(
        (
            "2B  stacking: +bf16 vs int4",
            {"delta_vs_fp16": stack["mean"], "ci95_vs_fp16": stack["ci95"]},
        )
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.62))
    fig.subplots_adjust(left=0.42, right=0.98, top=0.90, bottom=0.20)
    y = np.array([5.0, 4.0, 2.35, 1.35, 0.35])
    ax.axvline(0, color=GRAY, lw=0.8, ls=(0, (3, 2)), zorder=0)
    for yi, (label, row) in zip(y, entries):
        est = row["delta_vs_fp16"] * 100.0
        ci = [v * 100.0 for v in row["ci95_vs_fp16"]]
        excludes_zero = ci[0] > 0 or ci[1] < 0
        color = RED if excludes_zero else (BLUE if "state" in label else GRAY_DARK)
        draw_ci(
            ax,
            yi,
            est,
            ci,
            color=color,
            marker="s" if excludes_zero else "o",
            size=28,
        )
        align = "left" if est >= 0 else "right"
        dx = 0.20 if est >= 0 else -0.20
        ax.text(est + dx, yi + 0.20, fmt_signed(est, 2), color=color,
                fontsize=5.8, ha=align, va="bottom", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in entries])
    ax.set_ylim(-0.15, 5.65)
    ax.set_xlim(-5.1, 3.5)
    ax.set_xlabel("GSM8K accuracy change (percentage points)")
    ax.text(-5.05, 5.55, "State precision", color=BLUE, fontsize=6.2,
            fontweight="bold", va="bottom")
    ax.text(-5.05, 2.90, "KV and stacking controls", color=GRAY_DARK,
            fontsize=6.2, fontweight="bold", va="bottom")
    ax.text(3.45, 5.58, "1,800 paired draws\nitem + seed clustered", color=GRAY_DARK,
            fontsize=5.4, va="top", ha="right", linespacing=1.0)
    ax.text(3.45, -0.05, "square: 95% CI excludes zero", color=GRAY_DARK,
            fontsize=5.4, va="bottom", ha="right")
    clean_axis(ax)
    save_vector(fig, "fig2_gsm8k")
    plt.close(fig)


def fig3_ppl_ruler() -> None:
    ppl = load("results/quality/ppl-stacking-analysis-20260809.json")
    ruler = load(
        "results/reproduction/2026-08-13/ruler-nothink/"
        "ruler-nothink-5cell-gate4-20260813/gate4_validation.json"
    )

    fig = plt.figure(figsize=(3.35, 3.75))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 2.15], hspace=0.62,
                          left=0.36, right=0.98, top=0.95, bottom=0.13)

    ax = fig.add_subplot(gs[0, 0])
    ppl_rows = [("C4", ppl["tables"]["c4"]), ("PG19", ppl["tables"]["pg19"])]
    y = [1, 0]
    ax.axvline(0, color=GRAY, lw=0.8, ls=(0, (3, 2)))
    for yi, (name, row) in zip(y, ppl_rows):
        est, ci = row["delta_bf16_vs_fp32"], row["ci95_delta"]
        draw_ci(ax, yi, est, ci, color=BLUE, size=26)
        ax.text(ci[1] + 0.003, yi, fmt_signed(est, 4), fontsize=5.7,
                color=BLUE, va="center", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ppl_rows])
    ax.set_xlim(-0.055, 0.067)
    ax.set_ylim(-0.45, 1.45)
    ax.set_xlabel("Perplexity change (bf16 - fp32 state)")
    ax.set_title("PPL stacking under int4 KV (3 seeds)", loc="left", pad=4,
                 color=INK, fontweight="bold")
    panel_label(ax, "a", x=-0.34, y=1.05)
    clean_axis(ax)

    ax = fig.add_subplot(gs[1, 0])
    rows = []
    for row in ruler["statistical_findings"]:
        task = "FWE" if "fwe" in row["task"] else "NIAH multi-query"
        rows.append((f"{row['model'].upper()}  {task}  {row['length'] // 1024}K",
                     row["mean_delta_accuracy_points"],
                     row["ci95_delta_accuracy_points"]))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=GRAY, lw=0.8, ls=(0, (3, 2)))
    for yi, (_, est, ci) in zip(y, rows):
        draw_ci(ax, yi, est, ci, color=TEAL, size=26)
        ax.text(0.18, yi, "0.00", fontsize=5.6, color=TEAL,
                va="center", ha="left", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlabel("RULER accuracy change (percentage points)")
    ax.set_title("RULER no-think (3 paired seeds)", loc="left",
                 pad=4, color=INK, fontweight="bold")
    ax.text(0.99, 0.03, "Exact rerun: 30/30\nNot an equivalence test",
            transform=ax.transAxes, linespacing=1.2,
            color=GRAY_DARK, fontsize=5.5, ha="right", va="bottom")
    panel_label(ax, "b", x=-0.34, y=1.05)
    clean_axis(ax)

    save_vector(fig, "fig3_ppl_ruler")
    plt.close(fig)


def fig4_serving() -> None:
    formal = load("results/verified/2026-08-09/statebf16-serving-formal-analysis.json")
    second = load("results/verified/2026-08-09/statebf16-serving-repro-analysis.json")
    direction = load(
        "results/quality/serving-direction/serving-direction-agreement-20260811.json"
    )
    gate4 = load(
        "results/reproduction/2026-08-13/m4-four-config/gate4-r3/"
        "m4_gate4_validation.json"
    )

    def pick(js: dict, rate: int, threshold: int) -> dict:
        return next(
            r for r in js["paired_deltas"]
            if r["workload"] == "random" and r["rate"] == rate
            and r["threshold_ms"] == threshold
        )

    fig = plt.figure(figsize=(7.0, 5.25))
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.65, 1.0],
        height_ratios=[2.45, 1.0],
        left=0.055,
        right=0.995,
        top=0.91,
        bottom=0.10,
        wspace=0.34,
        hspace=0.55,
    )
    left = outer[0, 0].subgridspec(1, 3, wspace=0.12)
    rates = [40, 45, 50]
    thresholds = [250, 500, 1000, 2000, 3000]
    x = np.arange(len(thresholds))
    axes = []
    for i, rate in enumerate(rates):
        ax = fig.add_subplot(left[0, i], sharey=axes[0] if axes else None)
        axes.append(ax)
        frows = [pick(formal, rate, t) for t in thresholds]
        srows = [pick(second, rate, t) for t in thresholds]
        for rows, offset, color, marker in (
            (frows, -0.10, NAVY, "o"),
            (srows, +0.10, BLUE, "D"),
        ):
            means = np.array([r["mean_delta_goodput"] for r in rows])
            lo = np.array([r["ci95"][0] for r in rows])
            hi = np.array([r["ci95"][1] for r in rows])
            ax.errorbar(
                x + offset,
                means,
                yerr=np.vstack([means - lo, hi - means]),
                fmt=marker,
                ms=3.6,
                lw=0,
                elinewidth=1.0,
                capsize=1.8,
                color=color,
                markerfacecolor=color,
                markeredgecolor=WHITE,
                markeredgewidth=0.45,
                zorder=3,
            )
        ax.axhline(0, color=GRAY, lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(["250", "500", "1k", "2k", "3k"])
        ax.set_title(f"{rate} req/s", fontsize=6.7, fontweight="bold", pad=3)
        ax.set_xlim(-0.45, 4.45)
        ax.set_ylim(-0.15, 0.68)
        ax.set_xlabel("TTFT threshold (ms)")
        if i == 0:
            ax.set_ylabel("Paired goodput change (req/s)\n(bf16 - fp32 state)")
        else:
            ax.tick_params(labelleft=False)
            ax.spines["left"].set_visible(False)
        clean_axis(ax)

    axes[0].text(0.0, 1.22, "Random60 overload region", transform=axes[0].transAxes,
                 fontsize=7.2, fontweight="bold", color=INK, ha="left")
    axes[0].legend(
        handles=[
            Line2D([], [], marker="o", color=NAVY, lw=0, label="formal", ms=4),
            Line2D([], [], marker="D", color=BLUE, lw=0, label="temporal rerun", ms=4),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.15),
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0,
    )
    panel_label(axes[0], "a", x=-0.22, y=1.18)

    # Boundary matrix uses patches so the PDF/SVG stays fully vector.
    ax = fig.add_subplot(outer[0, 1])
    workloads = [("Random60", "random"), ("ShareGPT300", "sharegpt")]
    matrix_rows: list[tuple[str, list[float]]] = []
    for display, key in workloads:
        for run_name, js in (("formal", formal), ("temporal", second)):
            vals = []
            for t in thresholds:
                base = js["boundaries"][f"int4__{key}"][str(t)]
                bf16 = js["boundaries"][f"int4_statebf16__{key}"][str(t)]
                vals.append(float(bf16 - base))
            matrix_rows.append((f"{display} - {run_name}", vals))

    nrow, ncol = len(matrix_rows), len(thresholds)
    for row_idx, (_, vals) in enumerate(matrix_rows):
        y = nrow - 1 - row_idx
        for col_idx, value in enumerate(vals):
            if value == 0:
                fc, ec = TEAL_LIGHT, TEAL
            elif value > 0:
                fc, ec = ORANGE_LIGHT, ORANGE
            else:
                fc, ec = "#F4D8D2", RED
            ax.add_patch(Rectangle((col_idx - 0.43, y - 0.38), 0.86, 0.76,
                                   facecolor=fc, edgecolor=ec, lw=0.8))
            ax.text(col_idx, y, fmt_signed(value, 0), ha="center", va="center",
                    fontsize=6.0, color=INK, fontweight="bold")

    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(-0.65, nrow - 0.35)
    ax.set_xticks(np.arange(ncol))
    ax.set_xticklabels(["250", "500", "1k", "2k", "3k"])
    ax.set_yticks(np.arange(nrow)[::-1])
    ax.set_yticklabels([r[0] for r in matrix_rows])
    ax.set_xlabel("TTFT threshold (ms)")
    ax.set_title("Sustainable-rate boundary change (req/s)", loc="left",
                 fontsize=7.0, fontweight="bold", pad=22)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    qn = direction["input_summary"]["bh_formal_n_q_lt_0_05"]
    total = direction["input_summary"]["n_cells_per_run"]
    ax.text(
        0.5,
        1.055,
        f"cell = bf16 boundary - fp32 boundary; BH-FDR: {qn}/{total} cells with q < 0.05",
        transform=ax.transAxes,
        fontsize=5.7,
        color=GRAY_DARK,
        ha="center",
    )
    panel_label(ax, "b", x=-0.22, y=1.18)

    # The frozen four-configuration audit is the primary run-stability gate.
    ax = fig.add_subplot(outer[1, :])
    comparison = gate4["comparison"]
    gate_rows = [
        (
            "Continuous goodput\n(primary gate)",
            int(comparison["within_tolerance"]),
            int(comparison["outside_tolerance"]),
            "within 10%",
            "outside 10%",
            RED,
        ),
        (
            "SLO point labels\n(secondary check)",
            int(comparison["boundary_points_exact"]),
            int(comparison["boundary_points_total"])
            - int(comparison["boundary_points_exact"]),
            "exact",
            "changed",
            ORANGE,
        ),
        (
            "All-seed boundaries\n(secondary check)",
            int(comparison["boundaries_exact"]),
            int(comparison["boundaries_total"])
            - int(comparison["boundaries_exact"]),
            "exact",
            "changed",
            ORANGE,
        ),
    ]
    y_positions = np.arange(len(gate_rows))[::-1]
    for y, (_, stable, unstable, stable_label, unstable_label, alert) in zip(
        y_positions, gate_rows
    ):
        total_count = stable + unstable
        stable_pct = 100.0 * stable / total_count
        unstable_pct = 100.0 - stable_pct
        ax.barh(
            y,
            stable_pct,
            height=0.56,
            color=TEAL_LIGHT,
            edgecolor=TEAL,
            linewidth=0.8,
        )
        ax.barh(
            y,
            unstable_pct,
            left=stable_pct,
            height=0.56,
            color="#F4D8D2" if alert == RED else ORANGE_LIGHT,
            edgecolor=alert,
            linewidth=0.8,
        )
        ax.text(
            stable_pct / 2.0,
            y,
            f"{stable}/{total_count} {stable_label}",
            ha="center",
            va="center",
            fontsize=5.8,
            color=INK,
            fontweight="bold",
        )
        if unstable_pct >= 12.0:
            ax.text(
                stable_pct + unstable_pct / 2.0,
                y,
                f"{unstable}/{total_count} {unstable_label}",
                ha="center",
                va="center",
                fontsize=5.8,
                color=INK,
                fontweight="bold",
            )
        else:
            ax.text(
                101.5,
                y,
                f"{unstable}/{total_count} {unstable_label}",
                ha="left",
                va="center",
                fontsize=5.7,
                color=alert,
                fontweight="bold",
            )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row[0] for row in gate_rows])
    ax.set_xlim(0, 116)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Temporal comparisons (%)")
    ax.set_title(
        "Run-stability gate fails on continuous goodput",
        loc="left",
        fontsize=7.0,
        fontweight="bold",
        pad=5,
    )
    ax.text(
        1.0,
        1.08,
        "PRIMARY GATE: FAIL",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=RED,
        fontweight="bold",
    )
    ax.text(
        0.0,
        -0.43,
        "same seeds and host; temporal rerun is not independent replication",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color=GRAY_DARK,
    )
    ax.tick_params(axis="y", length=0)
    clean_axis(ax)
    panel_label(ax, "c", x=-0.06, y=1.10)

    save_vector(fig, "fig4_serving")
    plt.close(fig)


def fig5_block_granularity() -> None:
    cap = load("results/verified/2026-08-14/capacity-2x2-analysis-corrected.json")
    by = {(r["model"].upper(), r["length"], r["kv_dtype"]): r for r in cap["rows"]}
    cats = [
        ("2B", 4096, "int4"),
        ("2B", 16384, "int4"),
        ("9B", 4096, "int4"),
        ("9B", 16384, "int4"),
        ("2B", 4096, "fp16"),
        ("2B", 16384, "fp16"),
        ("9B", 4096, "fp16"),
    ]
    labels = [f"{kv} - {m} - {l // 1024}K" for m, l, kv in cats]
    y = np.arange(len(cats))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.87, bottom=0.18, wspace=0.20)
    specs = [
        ("fp32_block_size", "bf16_block_size", "Tokens per GPU block",
         "State bytes shrink each block"),
        ("fp32_num_gpu_blocks", "bf16_num_gpu_blocks", "Allocated GPU blocks",
         "Freed bytes become allocatable blocks"),
    ]
    for idx, (ax, (ka, kb, xlabel, title)) in enumerate(zip(axes, specs)):
        va = np.array([by[c][ka] for c in cats], dtype=float)
        vb = np.array([by[c][kb] for c in cats], dtype=float)
        for yi, a, b in zip(y, va, vb):
            ax.plot([a, b], [yi, yi], color=GRAY, lw=1.1, zorder=1)
            ax.scatter([a], [yi], s=28, color=GRAY_DARK, marker="o", zorder=2,
                       edgecolor=WHITE, linewidth=0.5)
            ax.scatter([b], [yi], s=31, color=BLUE, marker="o", zorder=3,
                       edgecolor=WHITE, linewidth=0.5)
        ax.set_yticks(y)
        if idx == 0:
            ax.set_yticklabels(labels)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        ax.axhline(2.5, color=GRAY_LIGHT, lw=0.8, zorder=0)
        clean_axis(ax)
        panel_label(ax, "a" if idx == 0 else "b", x=-0.16 if idx == 0 else -0.08,
                    y=1.03)

    axes[0].text(-0.31, 0.78, "int4 KV", transform=axes[0].transAxes,
                 color=ORANGE, fontsize=5.9, fontweight="bold", rotation=90,
                 rotation_mode="anchor", ha="center", va="center")
    axes[0].text(-0.31, 0.16, "fp16 KV", transform=axes[0].transAxes,
                 color=BLUE, fontsize=5.9, fontweight="bold", rotation=90,
                 rotation_mode="anchor", ha="center", va="center")
    axes[1].legend(
        handles=[
            Line2D([], [], marker="o", color="none", markerfacecolor=GRAY_DARK,
                   markeredgecolor=WHITE, ms=5, label="FP32 state"),
            Line2D([], [], marker="o", color="none", markerfacecolor=BLUE,
                   markeredgecolor=WHITE, ms=5, label="BF16 state"),
        ],
        loc="lower right",
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    save_vector(fig, "fig5_block_granularity")
    plt.close(fig)


def fig6_sensitivity() -> None:
    js = load("results/quality/state-sensitivity-analysis-20260809-bonf.json")
    rows = [r for r in js["rows"] if r["config"].startswith("bf16_L")]
    layers = [int(r["config"].split("L")[1]) for r in rows]
    y = np.arange(len(rows))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.28), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.15, wspace=0.12)
    for idx, (tag, title, ax) in enumerate((
        ("c4", "C4", axes[0]),
        ("pg19", "PG19", axes[1]),
    )):
        ax.axvline(0, color=GRAY, lw=0.8, ls=(0, (3, 2)), zorder=0)
        for yi, row in zip(y, rows):
            est = row[f"{tag}_delta"]
            ci = row[f"{tag}_ci95"]
            raw = bool(row.get(f"{tag}_sensitive", False))
            draw_ci(ax, yi, est, ci, color=GRAY_DARK, size=24)
            if raw:
                ax.scatter([est], [yi], s=50, facecolor="none", edgecolor=ORANGE,
                           linewidth=1.0, zorder=4)
        ax.set_yticks(y)
        if idx == 0:
            ax.set_yticklabels(layers)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("Per-layer PPL change (bf16 layer - fp32)")
        ax.set_title(f"{title} (3 paired seeds)", loc="left", fontweight="bold", pad=4)
        if idx == 0:
            ax.set_ylabel("GDN layer switched to bf16")
        clean_axis(ax)
        panel_label(ax, "a" if idx == 0 else "b", x=-0.12 if idx == 0 else -0.10,
                    y=1.01)

    fig.text(0.50, 0.95, "No per-layer effect survives Bonferroni or BH-FDR correction",
             ha="center", va="top", fontsize=7.1, fontweight="bold", color=INK)
    fig.text(0.50, 0.90, "orange ring: raw p < 0.05 only (2 of 36 tests)",
             ha="center", va="top", fontsize=5.8, color=ORANGE)
    save_vector(fig, "fig6_sensitivity")
    plt.close(fig)


def fig7_harness() -> None:
    chunk: dict[tuple[str, int], float] = {}
    for state in ("fp32", "bf16"):
        for size in (128, 1):
            path = ROOT / (
                "results/quality/chunk-ablation/"
                f"chunk-ablation-20260809__state{state}__chunk{size}__2b.csv"
            )
            with path.open(encoding="utf-8", newline="") as handle:
                chunk[(state, size)] = float(list(csv.DictReader(handle))[0]["ppl_mean"])
    ppl = load("results/quality/ppl-stacking-analysis-20260809.json")
    cost = ppl["stacking_cost_vs_fp16_kv"]

    fig = plt.figure(figsize=(3.35, 3.18))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 0.95], hspace=0.70,
                          left=0.20, right=0.98, top=0.95, bottom=0.16)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(2)
    width = 0.30
    fp32 = [chunk[("fp32", 128)], chunk[("fp32", 1)]]
    bf16 = [chunk[("bf16", 128)], chunk[("bf16", 1)]]
    ax.bar(x - width / 2, fp32, width, color=GRAY_DARK, ec=INK, lw=0.45,
           label="FP32 state")
    ax.bar(x + width / 2, bf16, width, color=BLUE, ec=INK, lw=0.45,
           label="BF16 state")
    ax.set_xticks(x)
    ax.set_xticklabels(["chunk = 128", "chunk = 1"])
    ax.set_ylabel("C4 perplexity\n(1 seed x 1 sequence)")
    ax.set_ylim(0, max(fp32 + bf16) * 1.23)
    pct = (fp32[1] / fp32[0] - 1.0) * 100.0
    ax.annotate(
        f"+{pct:.0f}%",
        xy=(1 - width / 2, fp32[1]),
        xytext=(0.56, fp32[1] + 4.0),
        textcoords="data",
        color=RED,
        fontsize=6.3,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 0.8},
        ha="center",
    )
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.35)
    ax.set_title("Chunk-level write-back changes the PPL scale", loc="left",
                 fontweight="bold", pad=4)
    panel_label(ax, "a", x=-0.20, y=1.04)
    clean_axis(ax)

    ax = fig.add_subplot(gs[1, 0])
    corpora = ["C4", "PG19"]
    fp16_vals = np.array([cost[c.lower()]["fp16kv_state_delta"] for c in corpora])
    int4_vals = np.array([cost[c.lower()]["int4kv_state_delta"] for c in corpora])
    y = np.array([1, 0])
    ax.axvline(0, color=GRAY, lw=0.8, ls=(0, (3, 2)))
    for yi, a, b in zip(y, fp16_vals, int4_vals):
        ax.plot([a, b], [yi, yi], color=GRAY, lw=1.1)
        ax.scatter([a], [yi], s=26, color=GRAY_DARK, edgecolor=WHITE,
                   linewidth=0.5, zorder=2)
        ax.scatter([b], [yi], s=28, color=ORANGE, edgecolor=WHITE,
                   linewidth=0.5, zorder=3)
        ax.text(a, yi + 0.18, fmt_signed(a, 4), ha="center", va="bottom",
                fontsize=5.5, color=GRAY_DARK)
        ax.text(b, yi - 0.18, fmt_signed(b, 4), ha="center", va="top",
                fontsize=5.5, color=ORANGE)
    ax.set_yticks(y)
    ax.set_yticklabels(corpora)
    ax.set_xlabel("Marginal state-bf16 perplexity change")
    ax.set_ylim(-0.50, 1.50)
    ax.set_title("Stacking cost remains small under either KV dtype", loc="left",
                 fontweight="bold", pad=4)
    ax.set_xlim(-0.0034, 0.0070)
    ax.text(0.99, 0.95, "gray: fp16 KV   orange: int4 KV",
            transform=ax.transAxes, fontsize=5.4, color=GRAY_DARK,
            ha="right", va="top")
    panel_label(ax, "b", x=-0.20, y=1.04)
    clean_axis(ax)

    save_vector(fig, "fig7_harness")
    plt.close(fig)


def fig8_gsm8k_per_seed() -> None:
    g2b = load("results/quality/gsm8k-state9seed-v2-dependence-aware-20260814.json")
    g9b = load("results/quality/gsm8k-9b-state9seed-v2-dependence-aware-20260814.json")

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.85))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.16, hspace=0.58)

    def draw(ax: mpl.axes.Axes, rows: list[dict], allocs: list[str], labels: list[str],
             title: str, ypad: float) -> None:
        by = {r["allocation"]: r for r in rows}
        seeds = list(by[allocs[0]]["per_seed"].keys())
        x = np.arange(len(allocs))
        all_values = []
        for seed in seeds:
            values = np.array([by[a]["per_seed"][seed] for a in allocs]) * 100.0
            all_values.extend(values.tolist())
            ax.plot(x, values, color=GRAY, lw=0.65, alpha=0.72, zorder=1)
        means = np.array(
            [by[a]["mean_accuracy_over_seed_item_draws"] for a in allocs]
        ) * 100.0
        ax.plot(x, means, color=BLUE, lw=1.7, marker="o", ms=4.0,
                markeredgecolor=WHITE, markeredgewidth=0.5, zorder=3)
        for xi, value in zip(x, means):
            ax.text(xi, value + ypad * 0.16, f"{value:.1f}", ha="center",
                    va="bottom", fontsize=5.6, color=BLUE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        low, high = min(all_values), max(all_values)
        ax.set_ylim(low - ypad, high + ypad)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        clean_axis(ax)

    draw(
        axes[0],
        g2b["rows"],
        ["fp16", "fp16_statebf16", "uniform_int4", "uniform_int4_statebf16"],
        ["fp16 KV\nfp32 state", "fp16 KV\nbf16 state",
         "int4 KV\nfp32 state", "int4 KV\nbf16 state"],
        "2B - descriptive trajectories across 9 dataset seeds",
        2.1,
    )
    draw(
        axes[1],
        g9b["rows"],
        ["fp16", "fp16_statebf16"],
        ["fp16 KV\nfp32 state", "fp16 KV\nbf16 state"],
        "9B - descriptive trajectories across 9 dataset seeds",
        1.2,
    )
    axes[0].text(0.99, 0.04, "gray: seed summaries   blue: draw-weighted mean",
                 transform=axes[0].transAxes, ha="right", va="bottom",
                 fontsize=5.5, color=GRAY_DARK)
    panel_label(axes[0], "a", x=-0.18, y=1.04)
    panel_label(axes[1], "b", x=-0.18, y=1.04)

    save_vector(fig, "fig8_gsm8k_per_seed")
    plt.close(fig)


def main() -> None:
    fig1_capacity()
    fig2_gsm8k()
    fig3_ppl_ruler()
    fig4_serving()
    fig5_block_granularity()
    fig6_sensitivity()
    fig7_harness()
    fig8_gsm8k_per_seed()
    from make_top_venue_figures import main as make_top_venue_figures

    make_top_venue_figures()
    print(f"Vector figures written to {OUT}")
    print(f"QA previews written to {PREVIEW}")


if __name__ == "__main__":
    main()
