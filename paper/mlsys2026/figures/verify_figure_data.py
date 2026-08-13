"""Cross-check every number shown in the figures against the atomic JSONs.

Prints a machine-readable ledger of (figure, label, value, source). Values are
re-derived here, not copied from the plotting script, so a mismatch indicates
either a plotting bug or a source-data change.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEDGER: list[tuple[str, str, object, str]] = []


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def add(fig: str, label: str, value, source: str):
    LEDGER.append((fig, label, value, source))


# Figure 1: capacity.
cap = load("results/verified/2026-08-09/capacity-2x2-analysis.json")
for r in cap["rows"]:
    add("fig1", f"{r['model']} L{r['length']} {r['kv_dtype']} measured",
        r["measured_r_state"], "capacity-2x2-analysis.json")
    add("fig1", f"{r['model']} L{r['length']} {r['kv_dtype']} predicted",
        r["predicted_r_state"], "capacity-2x2-analysis.json")
    add("fig1", f"{r['model']} L{r['length']} {r['kv_dtype']} gap%",
        r["signed_gap_pct"], "capacity-2x2-analysis.json")
for r in cap["r_kv_rows"]:
    if r["model"] == "2b" and r["length"] == 4096:
        add("fig1", f"r_kv {r['state_dtype']} state",
            r["measured_r_kv"], "capacity-2x2-analysis.json")

# Figure 2: GSM8K.
g2b = load("results/quality/gsm8k-state9seed-v2-analysis-20260809.json")
g9b = load("results/quality/gsm8k-9b-state9seed-v2-analysis-20260809.json")
for r in g2b["rows"]:
    if "delta_vs_fp16" not in r:
        continue
    add("fig2", f"2B {r['allocation']} delta",
        r["delta_vs_fp16"], "gsm8k-state9seed-v2-analysis")
    add("fig2", f"2B {r['allocation']} p", r["paired_t"]["p_value"], "same")
add("fig2", "2B stacking delta", g2b["stacking_marginal"]["mean"],
    "gsm8k-state9seed-v2-analysis")
add("fig2", "2B stacking p", g2b["stacking_marginal"]["paired_t"]["p_value"],
    "same")
add("fig2", "9B state delta", g9b["rows"][1]["delta_vs_fp16"],
    "gsm8k-9b-state9seed-v2-analysis")
add("fig2", "9B state p", g9b["rows"][1]["paired_t"]["p_value"], "same")

# Figure 3: PPL + RULER.
ppl = load("results/quality/ppl-stacking-analysis-20260809.json")
for corpus, tab in ppl["tables"].items():
    add("fig3", f"PPL {corpus} delta", tab["delta_bf16_vs_fp32"],
        "ppl-stacking-analysis")
    add("fig3", f"PPL {corpus} CI", tab["ci95_delta"], "same")
rul = load(
    "results/reproduction/2026-08-13/ruler-nothink/"
    "ruler-nothink-5cell-gate4-20260813/gate4_validation.json"
)
for r in rul["statistical_findings"]:
    add("fig3", f"RULER {r['model']} {r['task']} L{r['length']} delta",
        r["mean_delta_accuracy_points"], "ruler-nothink Gate 4")
    add("fig3", f"RULER {r['model']} {r['task']} L{r['length']} CI",
        r["ci95_delta_accuracy_points"], "same")

# Figure 4: serving.
for tag in ("formal", "repro"):
    js = load(f"results/verified/2026-08-09/statebf16-serving-{tag}-analysis.json")
    for r in js["paired_deltas"]:
        if r["workload"] == "random" and r["rate"] >= 40:
            add("fig4", f"{tag} r{r['rate']} {r['threshold_ms']}ms delta",
                round(r["mean_delta_goodput"], 4),
                f"statebf16-serving-{tag}-analysis")
            add("fig4", f"{tag} r{r['rate']} {r['threshold_ms']}ms CI",
                [round(v, 4) for v in r["ci95"]], "same")
    for key in ("int4__random", "int4__sharegpt",
                "int4_statebf16__random", "int4_statebf16__sharegpt"):
        add("fig4", f"{tag} boundary {key}",
            js["boundaries"][key], "same")

# Figure 5: block granularity.
for r in cap["rows"]:
    add("fig5", f"{r['model']} L{r['length']} {r['kv_dtype']} fp32 block size",
        r["fp32_block_size"], "capacity-2x2-analysis.json")
    add("fig5", f"{r['model']} L{r['length']} {r['kv_dtype']} bf16 block size",
        r["bf16_block_size"], "capacity-2x2-analysis.json")
    add("fig5", f"{r['model']} L{r['length']} {r['kv_dtype']} fp32 blocks",
        r["fp32_num_gpu_blocks"], "capacity-2x2-analysis.json")
    add("fig5", f"{r['model']} L{r['length']} {r['kv_dtype']} bf16 blocks",
        r["bf16_num_gpu_blocks"], "capacity-2x2-analysis.json")

# Figure 6: per-layer sensitivity.
sens = load("results/quality/state-sensitivity-analysis-20260809-bonf.json")
for r in sens["rows"]:
    if not r["config"].startswith("bf16_L"):
        continue
    add("fig6", f"{r['config']} c4 delta", r["c4_delta"],
        "state-sensitivity-analysis-20260809-bonf.json")
    add("fig6", f"{r['config']} c4 ci", r["c4_ci95"], "same")
    add("fig6", f"{r['config']} pg19 delta", r["pg19_delta"], "same")
    add("fig6", f"{r['config']} pg19 ci", r["pg19_ci95"], "same")

# Figure 7: chunk ablation + stacking cost.
import csv as _csv
for st in ("fp32", "bf16"):
    for ck in (128, 1):
        p = ROOT / "results" / "quality" / "chunk-ablation" / (
            f"chunk-ablation-20260809__state{st}__chunk{ck}__2b.csv")
        with p.open(encoding="utf-8") as fh:
            val = float(list(_csv.DictReader(fh))[0]["ppl_mean"])
        add("fig7", f"chunk {st} {ck} ppl", val, p.name)
for corpus in ("c4", "pg19"):
    add("fig7", f"{corpus} fp16kv state delta",
        ppl["stacking_cost_vs_fp16_kv"][corpus]["fp16kv_state_delta"],
        "ppl-stacking-analysis")
    add("fig7", f"{corpus} int4kv state delta",
        ppl["stacking_cost_vs_fp16_kv"][corpus]["int4kv_state_delta"], "same")

# Figure 8: GSM8K per-seed means.
for r in g2b["rows"]:
    add("fig8", f"2B {r['allocation']} mean", r["mean"],
        "gsm8k-state9seed-v2-analysis")
for r in g9b["rows"]:
    add("fig8", f"9B {r['allocation']} mean", r["mean"],
        "gsm8k-9b-state9seed-v2-analysis")

for fig, label, value, source in LEDGER:
    print(f"{fig}\t{label}\t{value}\t{source}")
print(f"\nLEDGER_ENTRIES={len(LEDGER)}")
