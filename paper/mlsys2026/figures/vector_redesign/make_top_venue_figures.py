"""Claim-led main-paper figures for the MLSys 2026 submission.

System and mechanism schematics are authored in Next AI Draw.io. This module
draws only the quantitative evidence panels from frozen repository artifacts.
It intentionally reuses the export and typography contract from
``make_vector_figures.py`` so the main and supplementary figures remain one
visual system.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import make_vector_figures as base


BLUE = base.BLUE
BLUE_LIGHT = base.BLUE_LIGHT
TEAL = base.TEAL
TEAL_LIGHT = base.TEAL_LIGHT
ORANGE = base.ORANGE
ORANGE_LIGHT = base.ORANGE_LIGHT
RED = base.RED
GRAY_DARK = base.GRAY_DARK
GRAY = base.GRAY
GRAY_LIGHT = base.GRAY_LIGHT
INK = base.INK
WHITE = base.WHITE

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
    }
)


def save_vector(fig: mpl.figure.Figure, name: str) -> None:
    """Export the main-paper override with the shared publication contract."""

    base.OUT.mkdir(parents=True, exist_ok=True)
    base.PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        base.PREVIEW / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        base.PREVIEW / f"{name}.tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def fig1_capacity() -> None:
    """Full 52-pair capacity evidence for the right side of Figure 1."""

    cap = base.load(
        "results/verified/2026-08-14/"
        "capacity-phase-formal-corrected.analysis.json"
    )
    rows = cap["rows"]
    gains = np.array(
        [
            100.0 * (row["bf16_state_tokens"] / row["fp32_state_tokens"] - 1.0)
            for row in rows
        ]
    )
    residuals = np.array([row["prediction_residual_pct"] for row in rows])

    fig = plt.figure(figsize=(2.78, 2.67))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.28, 1.0],
        left=0.20,
        right=0.985,
        top=0.94,
        bottom=0.15,
        hspace=0.54,
    )

    ax = fig.add_subplot(gs[0, 0])
    order = np.argsort(gains)
    x = np.arange(1, len(rows) + 1)
    ordered_rows = [rows[index] for index in order]
    colors = [ORANGE if row["kv_dtype"] == "int4" else BLUE for row in ordered_rows]
    ax.plot(x, gains[order], color=GRAY_LIGHT, lw=0.75, zorder=1)
    ax.scatter(
        x,
        gains[order],
        c=colors,
        s=13,
        edgecolor=WHITE,
        linewidth=0.35,
        zorder=2,
    )
    median_gain = float(np.median(gains))
    ax.axhline(median_gain, color=GRAY_DARK, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(0, len(rows) + 1)
    ax.set_ylim(-2, 100)
    ax.set_xticks([])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Token-capacity gain (%)")
    ax.set_title(
        "Capacity increases in all 52 pairs",
        loc="left",
        fontweight="bold",
        pad=4,
    )
    ax.text(
        0.54,
        0.96,
        f"median +{median_gain:.2f}%\nrange +{gains.min():.2f} to +{gains.max():.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        color=INK,
        linespacing=1.25,
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color="none", markerfacecolor=ORANGE,
                   markeredgecolor=WHITE, ms=4, label="int4 KV"),
            Line2D([], [], marker="o", color="none", markerfacecolor=BLUE,
                   markeredgecolor=WHITE, ms=4, label="fp16 KV"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.82),
        ncol=1,
        borderaxespad=0,
        handletextpad=0.25,
        labelspacing=0.2,
    )
    base.clean_axis(ax)
    base.panel_label(ax, "b", x=-0.19, y=1.03)

    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(residuals)
    ax.plot(x, residuals[order], color=GRAY_LIGHT, lw=0.75, zorder=1)
    ax.scatter(
        x,
        residuals[order],
        color=GRAY_DARK,
        s=13,
        edgecolor=WHITE,
        linewidth=0.35,
        zorder=2,
    )
    ax.axhline(0, color=GRAY, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(0, len(rows) + 1)
    ax.set_ylim(-7.0, 15.0)
    ax.set_xticks([])
    ax.set_yticks([-5, 0, 5, 10, 15])
    ax.set_ylabel("Model residual (%)")
    ax.set_xlabel("Frozen allocator cells, ordered within each panel")
    ax.set_title(
        "Continuous model is approximate",
        loc="left",
        fontweight="bold",
        pad=4,
    )
    summary = cap["prediction_residual_summary"]
    ax.text(
        0.01,
        0.96,
        f"median |error| {summary['median_absolute_pct']:.4f}%\n"
        f"range {residuals.min():+.4f}% to {residuals.max():+.4f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=INK,
        linespacing=1.22,
    )
    base.clean_axis(ax)
    base.panel_label(ax, "c", x=-0.19, y=1.03)

    save_vector(fig, "fig1_capacity")
    plt.close(fig)


def fig2_gsm8k() -> None:
    """Dependence-aware quality intervals plus the fail-closed selector audit."""

    g2b = base.load(
        "results/quality/gsm8k-state9seed-v2-dependence-aware-20260814.json"
    )
    g9b = base.load(
        "results/quality/gsm8k-9b-state9seed-v2-dependence-aware-20260814.json"
    )
    selector = base.load(
        "results/verified/2026-08-14/controller-decisions/selector-audit.json"
    )
    rows2 = {row["allocation"]: row for row in g2b["rows"]}
    rows9 = {row["allocation"]: row for row in g9b["rows"]}
    entries = [
        ("2B  state bf16 vs fp32", rows2["fp16_statebf16"]),
        ("9B  state bf16 vs fp32", rows9["fp16_statebf16"]),
        ("2B  int4 KV vs fp16", rows2["uniform_int4"]),
        ("2B  int4 + bf16 vs fp16", rows2["uniform_int4_statebf16"]),
        (
            "2B  stacking: +bf16 vs int4",
            {
                "delta_vs_fp16": g2b["stacking_marginal"]["mean"],
                "ci95_vs_fp16": g2b["stacking_marginal"]["ci95"],
            },
        ),
    ]

    fig = plt.figure(figsize=(3.35, 3.62))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.62, 1.16],
        left=0.41,
        right=0.985,
        top=0.95,
        bottom=0.12,
        hspace=0.60,
    )

    ax = fig.add_subplot(gs[0, 0])
    y = np.array([5.0, 4.1, 2.45, 1.55, 0.65])
    ax.axvline(0, color=GRAY, lw=0.8, ls=(0, (3, 2)), zorder=0)
    for yi, (label, row) in zip(y, entries):
        estimate = row["delta_vs_fp16"] * 100.0
        interval = [value * 100.0 for value in row["ci95_vs_fp16"]]
        excludes_zero = interval[0] > 0 or interval[1] < 0
        color = RED if excludes_zero else (BLUE if "state" in label else GRAY_DARK)
        base.draw_ci(
            ax,
            yi,
            estimate,
            interval,
            color=color,
            marker="s" if excludes_zero else "o",
            size=26,
        )
        align = "left" if estimate >= 0 else "right"
        offset = 0.18 if estimate >= 0 else -0.18
        ax.text(
            estimate + offset,
            yi + 0.17,
            f"{estimate:+.2f}",
            color=color,
            fontsize=5.6,
            ha=align,
            va="bottom",
            fontweight="bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([entry[0] for entry in entries])
    ax.set_ylim(0.15, 5.60)
    ax.set_xlim(-5.35, 3.55)
    ax.set_xlabel("GSM8K accuracy change (percentage points)")
    ax.set_title(
        "Quality guardrail: candidate-specific 95% CIs",
        loc="left",
        fontweight="bold",
        pad=4,
    )
    ax.text(
        0.99,
        0.98,
        "1,800 paired draws\nitem + seed clustered",
        transform=ax.transAxes,
        fontsize=5.2,
        color=GRAY_DARK,
        ha="right",
        va="top",
        linespacing=1.05,
    )
    ax.text(
        0.99,
        0.02,
        "square: nominal 95% CI excludes zero",
        transform=ax.transAxes,
        fontsize=5.1,
        color=GRAY_DARK,
        ha="right",
        va="bottom",
    )
    base.clean_axis(ax)
    base.panel_label(ax, "a", x=-0.39, y=1.03)

    ax = fig.add_subplot(gs[1, 0])
    candidates = ["full", "kv_only", "state_only", "joint"]
    display = ["full", "KV only", "state only", "joint"]
    decisions = selector["decisions"]
    y_positions = np.arange(len(decisions))[::-1]
    reason_codes = {
        "insufficient_allocator_equivalent_sequence_slots": "C",
        "quality_guardrail_violated:gsm8k": "Q",
        "ttft_slo_violated": "T",
    }
    for y_value, decision in zip(y_positions, decisions):
        by_candidate = {row["candidate"]: row for row in decision["candidates"]}
        for x_value, candidate in enumerate(candidates):
            row = by_candidate[candidate]
            if row["selected"]:
                fill, edge, text_color, text = BLUE, BLUE, WHITE, "select"
            else:
                fill, edge, text_color = "#F8E4E0", RED, INK
                codes = [reason_codes[reason] for reason in row["rejection_reasons"]]
                text = "+".join(codes)
            ax.add_patch(
                Rectangle(
                    (x_value - 0.43, y_value - 0.34),
                    0.86,
                    0.68,
                    facecolor=fill,
                    edgecolor=edge,
                    lw=0.8,
                )
            )
            ax.text(
                x_value,
                y_value,
                text,
                ha="center",
                va="center",
                fontsize=5.5,
                color=text_color,
                fontweight="bold",
            )
    ax.set_xlim(-0.5, len(candidates) - 0.5)
    ax.set_ylim(-0.75, len(decisions) - 0.30)
    ax.set_xticks(np.arange(len(display)))
    ax.set_xticklabels(display)
    ax.xaxis.tick_top()
    ax.set_yticks(y_positions)
    ax.set_yticklabels([decision["label"] for decision in decisions])
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Fail-closed selector: 1 of 16 cells is feasible",
        loc="left",
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0.99,
        -0.18,
        "C: slots   Q: GSM8K   T: TTFT",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.2,
        color=GRAY_DARK,
    )
    base.panel_label(ax, "b", x=-0.39, y=1.08)

    save_vector(fig, "fig2_gsm8k")
    plt.close(fig)


def fig3_ppl_ruler() -> None:
    """Separate approximate PPL evidence from descriptive RULER equality."""

    ppl = base.load("results/quality/ppl-stacking-analysis-20260809.json")
    ruler = base.load(
        "results/reproduction/2026-08-13/ruler-nothink/"
        "ruler-nothink-5cell-gate4-20260813/gate4_validation.json"
    )

    fig = plt.figure(figsize=(3.35, 3.58))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.78, 1.65],
        hspace=0.64,
        left=0.37,
        right=0.985,
        top=0.95,
        bottom=0.14,
    )

    ax = fig.add_subplot(gs[0, 0])
    ppl_rows = [("C4", ppl["tables"]["c4"]), ("PG19", ppl["tables"]["pg19"])]
    y = [1, 0]
    ax.axvline(0, color=GRAY, lw=0.8, ls=(0, (3, 2)))
    for yi, (_, row) in zip(y, ppl_rows):
        estimate = row["delta_bf16_vs_fp32"]
        interval = row["ci95_delta"]
        base.draw_ci(ax, yi, estimate, interval, color=BLUE, size=25)
        ax.text(
            interval[1] + 0.003,
            yi,
            f"{estimate:+.4f}",
            fontsize=5.6,
            color=BLUE,
            va="center",
            ha="left",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in ppl_rows])
    ax.set_xlim(-0.055, 0.067)
    ax.set_ylim(-0.45, 1.45)
    ax.set_xlabel("Perplexity change (bf16 - fp32 state)")
    ax.set_title(
        "Chunk-level PPL under int4 KV (3 seeds)",
        loc="left",
        pad=4,
        fontweight="bold",
    )
    ax.text(
        0.99,
        0.98,
        "both CIs cross zero",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.2,
        color=GRAY_DARK,
    )
    base.panel_label(ax, "a", x=-0.34, y=1.05)
    base.clean_axis(ax)

    findings = ruler["statistical_findings"]
    match_counts: dict[tuple[str, str, int], tuple[int, int]] = {}
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for comparison in ruler["reproducibility"]["comparisons"]:
        cell = comparison["cell"]
        grouped[(cell["model"], cell["task"], int(cell["length"]))].append(comparison)
    for key, comparisons in grouped.items():
        match_counts[key] = (
            sum(comparison["status"] == "MATCH" for comparison in comparisons),
            len(comparisons),
        )

    labels = []
    matrix = []
    for row in findings:
        task = "FWE" if row["task"] == "ruler_fwe" else "NIAH multi-query"
        labels.append(f"{row['model'].upper()}  {task}  {row['length'] // 1024}K")
        matrix.append(
            (
                row["mean_delta_accuracy_points"],
                match_counts[(row["model"], row["task"], int(row["length"]))],
            )
        )

    ax = fig.add_subplot(gs[1, 0])
    y_positions = np.arange(len(labels))[::-1]
    for y_value, (delta, (matches, total)) in zip(y_positions, matrix):
        for x_value, text in enumerate((f"{delta:+.2f} pp", f"{matches}/{total} exact")):
            ax.add_patch(
                Rectangle(
                    (x_value - 0.43, y_value - 0.34),
                    0.86,
                    0.68,
                    facecolor=TEAL_LIGHT if x_value == 0 else "#EEF5F4",
                    edgecolor=TEAL,
                    lw=0.8,
                )
            )
            ax.text(
                x_value,
                y_value,
                text,
                ha="center",
                va="center",
                fontsize=5.5,
                color=INK,
                fontweight="bold",
            )
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.75, len(labels) - 0.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["observed state delta", "temporal rerun"])
    ax.xaxis.tick_top()
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "RULER observed equality and exact rerun",
        loc="left",
        pad=18,
        fontweight="bold",
    )
    ax.text(
        0.99,
        -0.17,
        "n = 3 paired seeds per cell; no equivalence margin",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.1,
        color=GRAY_DARK,
    )
    base.panel_label(ax, "b", x=-0.34, y=1.08)

    save_vector(fig, "fig3_ppl_ruler")
    plt.close(fig)


def fig5_block_granularity() -> None:
    """Generalize the Draw.io worked example across the seven measured cells."""

    cap = base.load("results/verified/2026-08-14/capacity-2x2-analysis-corrected.json")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in cap["rows"]:
        groups[(row["model"].upper(), row["kv_dtype"])].append(row)

    fig, ax = plt.subplots(figsize=(2.18, 2.55))
    fig.subplots_adjust(left=0.25, right=0.97, top=0.86, bottom=0.22)
    offsets = {
        ("2B", "int4"): (-5, -8),
        ("9B", "int4"): (5, 7),
        ("2B", "fp16"): (4, -2),
        ("9B", "fp16"): (6, 5),
    }
    for (model, kv_dtype), rows in sorted(groups.items()):
        row = rows[0]
        block_ratio = 100.0 * row["bf16_block_size"] / row["fp32_block_size"]
        count_ratio = row["bf16_num_gpu_blocks"] / row["fp32_num_gpu_blocks"]
        color = ORANGE if kv_dtype == "int4" else BLUE
        marker = "o" if model == "2B" else "D"
        ax.scatter(
            block_ratio,
            count_ratio,
            s=35 + 12 * (len(rows) - 1),
            marker=marker,
            color=color,
            edgecolor=WHITE,
            linewidth=0.65,
            zorder=3,
        )
        dx, dy = offsets[(model, kv_dtype)]
        ax.annotate(
            f"{model} {kv_dtype}\n{len(rows)} context{'s' if len(rows) > 1 else ''}",
            (block_ratio, count_ratio),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=5.8,
            color=color,
            fontweight="bold",
            linespacing=1.0,
        )
    ax.set_xlim(50.2, 53.55)
    ax.set_ylim(1.865, 1.995)
    ax.set_xticks([50.5, 51.5, 52.5, 53.5])
    ax.set_yticks([1.88, 1.92, 1.96, 2.00])
    ax.set_xlabel("BF16 / FP32 block size (%)")
    ax.set_ylabel("Allocated-block ratio\n(bf16 / fp32 state)")
    ax.set_title(
        "Discrete effect across 7 cells",
        loc="left",
        fontweight="bold",
        pad=5,
    )
    ax.text(
        0.02,
        0.02,
        "4 distinct layouts\nmarker: 2B circle, 9B diamond",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=GRAY_DARK,
        linespacing=1.05,
    )
    base.clean_axis(ax)
    base.panel_label(ax, "b", x=-0.23, y=1.04)

    save_vector(fig, "fig5_block_granularity")
    plt.close(fig)


def main() -> None:
    fig1_capacity()
    fig2_gsm8k()
    fig3_ppl_ruler()
    base.fig4_serving()
    fig5_block_granularity()
    print(f"Top-venue main-paper figures written to {base.OUT}")


if __name__ == "__main__":
    main()
