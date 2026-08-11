#!/usr/bin/env python3
"""E-7: paired Cohen's d for the GSM8K 9-seed quality comparisons.

Pure analysis over already-verified atomic JSON artifacts. No new experiment
is run, no historical data file is modified.

Definition: paired Cohen's d = mean(seed deltas) / SD(seed deltas), where each
seed delta is the per-seed accuracy difference between the two allocations
using the same 200-item subset. SD is the sample standard deviation (ddof=1).
2B and 9B are never pooled.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scipy import stats


ROOT = Path(r"E:\MLSys_Research")
OUT_DIR = ROOT / "results" / "quality" / "reasoning"
ANALYSIS_ID = "gsm8k-cohens-d-9seed-20260811"

INPUTS = {
    "gsm8k_2b": ROOT / "results" / "quality" / "gsm8k-state9seed-v2-analysis-20260809.json",
    "gsm8k_9b": ROOT / "results" / "quality" / "gsm8k-9b-state9seed-v2-analysis-20260809.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def compute_comparison(rows: dict[str, dict], base: str, target: str) -> dict:
    if base not in rows or target not in rows:
        return None
    per_seed_base = rows[base]["per_seed"]
    per_seed_target = rows[target]["per_seed"]
    if per_seed_base.keys() != per_seed_target.keys():
        raise SystemExit(f"seed keys differ for {base} vs {target}")
    keys = sorted(per_seed_base, key=lambda s: int(s))
    deltas = [per_seed_target[k] - per_seed_base[k] for k in keys]
    mean_delta = statistics.mean(deltas)
    sd_delta = statistics.stdev(deltas)
    cohens_d = mean_delta / sd_delta if sd_delta else None
    t_stat, p_value = stats.ttest_rel(
        [per_seed_target[k] for k in keys], [per_seed_base[k] for k in keys]
    )
    return {
        "base_allocation": base,
        "target_allocation": target,
        "archived_compare": f"{target} vs {base}",
        "n_seed_pairs": len(keys),
        "per_seed_delta": {k: round(deltas[i], 6) for i, k in enumerate(keys)},
        "mean_delta": round(mean_delta, 6),
        "sd_delta": round(sd_delta, 6),
        "paired_cohens_d": round(cohens_d, 6) if cohens_d is not None else None,
        "paired_ttest_p": round(float(p_value), 6),
        "archived_mean_delta": rows[target].get("delta_vs_fp16"),
        "archived_ci95": rows[target].get("ci95_vs_fp16"),
        "archived_p_value": (
            rows[target].get("paired_t", {}).get("p_value") if target != base else None
        ),
    }


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_hashes = {key: sha256_file(path) for key, path in INPUTS.items()}
    data = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in INPUTS.items()}

    comparisons = {
        "2b": [
            compute_comparison({r["allocation"]: r for r in data["gsm8k_2b"]["rows"]}, "fp16", "fp16_statebf16"),
            compute_comparison({r["allocation"]: r for r in data["gsm8k_2b"]["rows"]}, "fp16", "uniform_int4"),
            compute_comparison({r["allocation"]: r for r in data["gsm8k_2b"]["rows"]}, "uniform_int4", "uniform_int4_statebf16"),
        ],
        "9b": [
            compute_comparison({r["allocation"]: r for r in data["gsm8k_9b"]["rows"]}, "fp16", "fp16_statebf16"),
        ],
    }
    if any(c is None for c in comparisons["2b"] + comparisons["9b"]):
        raise SystemExit("missing allocation rows in GSM8K analysis inputs")
    stacking = data["gsm8k_2b"].get("stacking_marginal")
    if stacking:
        comparisons["2b"][2].update(
            {
                "archived_compare": stacking.get("compare"),
                "archived_mean_delta": stacking.get("mean"),
                "archived_ci95": stacking.get("ci95"),
                "archived_p_value": stacking.get("paired_t", {}).get("p_value"),
            }
        )

    created_at = "2026-08-11T00:00:00+08:00"
    contract = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "created_at": created_at,
        "status": "FROZEN",
        "git_commit": git_head(),
        "purpose": "Report paired Cohen's d for GSM8K 9-seed paired-accuracy comparisons.",
        "inputs": {key: {"path": str(path), "sha256": h} for key, (path, h) in
                   ((k, (INPUTS[k], input_hashes[k])) for k in INPUTS)},
        "definition": "paired Cohen's d = mean(seed deltas) / sample_SD(seed deltas)",
        "seed_semantics": (
            "each seed selects the same 200-item subset in every allocation; "
            "deltas are paired within seed"
        ),
        "pooling_rule": "2B and 9B are never pooled; int4 comparisons are reported separately",
        "limitations": [
            "n=9 paired seeds; effect-size precision is limited by the small seed count",
            "Cohen's d is a point estimate; the paper's CIs and power are reported alongside it",
            "no correction for the multiple quality comparisons in this file; the paper already discloses per-comparison p-values",
        ],
        "outputs": {
            "analysis_json": str(OUT_DIR / f"{ANALYSIS_ID}.json"),
            "contract_json": str(OUT_DIR / f"{ANALYSIS_ID}.contract.json"),
        },
    }
    contract_path = OUT_DIR / f"{ANALYSIS_ID}.contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_DIR / f"{ANALYSIS_ID}.contract.json.sha256").write_text(
        sha256_file(contract_path) + "\n", encoding="ascii"
    )

    self_review = [
        {"fallacy": "Simpson's paradox", "resolution": "2B and 9B are reported separately; stacking vs main effects are separate rows."},
        {"fallacy": "Ecological fallacy", "resolution": "Deltas are paired within seed at the model level; no per-item inference."},
        {"fallacy": "Berkson's paradox", "resolution": "All comparisons with available rows are included, including non-significant stacking and 9B rows."},
        {"fallacy": "Collider bias", "resolution": "No selection on significance; n=9 rows include zero-delta seeds."},
        {"fallacy": "Base-rate neglect", "resolution": "Effect sizes are reported with raw point deltas and CIs, not as standalone d values."},
        {"fallacy": "Regression to the mean", "resolution": "9B positive d is not claimed as a real gain because the paired CI contains zero."},
        {"fallacy": "Survivorship bias", "resolution": "No seeds or allocations are dropped."},
        {"fallacy": "Look-elsewhere effect", "resolution": "The three 2B rows and one 9B row are the paper's existing comparisons; no new windows are added."},
        {"fallacy": "Garden of forking paths", "resolution": "Formula and pooling rule are frozen in the contract before output generation."},
        {"fallacy": "Correlation != causation", "resolution": "Effect size describes paired accuracy difference; no mechanism claim."},
        {"fallacy": "Reverse causality", "resolution": "Allocation is the manipulated variable; seed accuracy is the outcome."},
    ]

    analysis = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "created_at": created_at,
        "contract_sha256": sha256_file(contract_path),
        "comparisons": comparisons,
        "self_review": self_review,
    }
    analysis_path = OUT_DIR / f"{ANALYSIS_ID}.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_DIR / f"{ANALYSIS_ID}.json.sha256").write_text(
        sha256_file(analysis_path) + "\n", encoding="ascii"
    )
    print(f"wrote {analysis_path}")
    print(f"wrote {contract_path}")


if __name__ == "__main__":
    run()
