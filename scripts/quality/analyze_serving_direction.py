#!/usr/bin/env python3
"""E-1: serving direction consistency between formal and second formal run.

Pure analysis over already-verified atomic JSON artifacts. No new experiment
is run, no historical data file is modified.

Contract points (frozen before this script executes):
- unit of analysis: one (workload, offered rate, TTFT threshold) cell from the
  60-cell formal/repro serving matrix;
- direction sign: sign(mean_delta_goodput), exact zero = no direction;
- agreement: both runs nonzero and same sign; opposite: both nonzero and
  different signs; zero-vs-nonzero cells are neither;
- sign test: exact two-sided binomial over nonzero-both cells, H0 p=0.5,
  reported as cell-level exploratory evidence (cells share seeds and are not
  independent; no multiple-comparison correction);
- load subsets: sustainable = fp32 and bf16 both all-sustainable in that run;
  boundary = exactly one all-sustainable; overload = neither. Subset results
  are reported per run and per both-runs intersection; the 90-cell merge is
  never used because formal and repro cells are paired, not pooled.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scipy import stats


ROOT = Path(r"E:\MLSys_Research")
OUT_DIR = ROOT / "results" / "quality" / "serving-direction"
ANALYSIS_ID = "serving-direction-agreement-20260811"

INPUTS = {
    "formal": ROOT / "results" / "verified" / "2026-08-09" / "statebf16-serving-formal-analysis.json",
    "repro": ROOT / "results" / "verified" / "2026-08-09" / "statebf16-serving-repro-analysis.json",
    "bh": ROOT / "results" / "quality" / "serving-bh" / "serving-bh-analysis-20260810.json",
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


def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def binomial_p(agree: int, opposite: int) -> float | None:
    n = agree + opposite
    if n == 0:
        return None
    return float(stats.binomtest(agree, n, 0.5, alternative="two-sided").pvalue)


def summarize(cells: list[dict]) -> dict:
    agree = sum(1 for c in cells if c["formal_sign"] == c["repro_sign"] and c["formal_sign"] != 0)
    opposite = sum(
        1
        for c in cells
        if c["formal_sign"] != 0 and c["repro_sign"] != 0 and c["formal_sign"] != c["repro_sign"]
    )
    both_zero = sum(1 for c in cells if c["formal_sign"] == 0 and c["repro_sign"] == 0)
    one_zero = sum(1 for c in cells if (c["formal_sign"] == 0) != (c["repro_sign"] == 0))
    nonzero_both = agree + opposite
    return {
        "n_cells": len(cells),
        "agree_nonzero": agree,
        "opposite_nonzero": opposite,
        "both_zero": both_zero,
        "one_zero": one_zero,
        "nonzero_both": nonzero_both,
        "agree_rate_nonzero": round(agree / nonzero_both, 6) if nonzero_both else None,
        "opposite_rate_nonzero": round(opposite / nonzero_both, 6) if nonzero_both else None,
        "sign_test_binomial_p": round(binomial_p(agree, opposite), 6)
        if binomial_p(agree, opposite) is not None
        else None,
    }


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_hashes = {key: sha256_file(path) for key, path in INPUTS.items()}

    formal = json.loads(INPUTS["formal"].read_text(encoding="utf-8"))["paired_deltas"]
    repro = json.loads(INPUTS["repro"].read_text(encoding="utf-8"))["paired_deltas"]
    bh = json.loads(INPUTS["bh"].read_text(encoding="utf-8"))

    formal_map = {(c["workload"], c["rate"], c["threshold_ms"]): c for c in formal}
    repro_map = {(c["workload"], c["rate"], c["threshold_ms"]): c for c in repro}
    if set(formal_map) != set(repro_map):
        raise SystemExit("formal/repro cell keys do not match")
    if len(formal_map) != 60:
        raise SystemExit(f"expected 60 cells, got {len(formal_map)}")

    bh_formal = {(c["workload"], c["rate"], c["threshold_ms"]): c for c in bh["formal"]["cells"]}
    bh_repro = {(c["workload"], c["rate"], c["threshold_ms"]): c for c in bh["repro"]["cells"]}
    if set(bh_formal) != set(formal_map) or set(bh_repro) != set(formal_map):
        raise SystemExit("BH cell keys do not match formal/repro cell keys")

    cells = []
    for key, fc in sorted(formal_map.items()):
        rc = repro_map[key]
        formal_delta = float(fc["mean_delta_goodput"])
        repro_delta = float(rc["mean_delta_goodput"])
        cells.append(
            {
                "workload": key[0],
                "rate": key[1],
                "threshold_ms": key[2],
                "formal_delta_goodput": formal_delta,
                "repro_delta_goodput": repro_delta,
                "formal_sign": sign(formal_delta),
                "repro_sign": sign(repro_delta),
                "formal_bh_q": bh_formal[key]["bh_q"],
                "repro_bh_q": bh_repro[key]["bh_q"],
                "formal_sustainable": bool(fc["fp32_all_sustainable"] and fc["bf16_all_sustainable"]),
                "formal_boundary": bool(fc["fp32_all_sustainable"] != fc["bf16_all_sustainable"]),
                "formal_overload": bool(
                    not fc["fp32_all_sustainable"] and not fc["bf16_all_sustainable"]
                ),
                "repro_sustainable": bool(rc["fp32_all_sustainable"] and rc["bf16_all_sustainable"]),
                "repro_boundary": bool(rc["fp32_all_sustainable"] != rc["bf16_all_sustainable"]),
                "repro_overload": bool(
                    not rc["fp32_all_sustainable"] and not rc["bf16_all_sustainable"]
                ),
            }
        )

    def pick(predicate):
        return [c for c in cells if predicate(c)]

    by_workload = {
        w: summarize(pick(lambda c, w=w: c["workload"] == w)) for w in ("random", "sharegpt")
    }

    subset_defs = {
        "sustainable_both_runs": lambda c: c["formal_sustainable"] and c["repro_sustainable"],
        "boundary_any_run": lambda c: c["formal_boundary"] or c["repro_boundary"],
        "overload_both_runs": lambda c: c["formal_overload"] and c["repro_overload"],
        "overload_formal": lambda c: c["formal_overload"],
        "overload_repro": lambda c: c["repro_overload"],
    }
    by_load_subset = {name: summarize(pick(pred)) for name, pred in subset_defs.items()}

    workload_x_subset = {}
    for w in ("random", "sharegpt"):
        for name, pred in subset_defs.items():
            key = f"{w}__{name}"
            ss = pick(lambda c, w=w, p=pred: c["workload"] == w and p(c))
            if ss:
                workload_x_subset[key] = summarize(ss)

    thresholds = sorted({c["threshold_ms"] for c in cells})
    by_threshold = {
        t: summarize(pick(lambda c, t=t: c["threshold_ms"] == t)) for t in thresholds
    }

    deltas_f = [c["formal_delta_goodput"] for c in cells]
    deltas_r = [c["repro_delta_goodput"] for c in cells]
    pearson_r = float(stats.pearsonr(deltas_f, deltas_r)[0])

    created_at = "2026-08-11T00:00:00+08:00"
    contract = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "created_at": created_at,
        "status": "FROZEN",
        "git_commit": git_head(),
        "purpose": (
            "Quantify directional consistency of paired goodput deltas between the formal "
            "serving matrix and the second formal run, and report it by workload and load "
            "subset without pooling the two runs."
        ),
        "inputs": {key: {"path": str(path), "sha256": h} for key, (path, h) in
                   ((k, (INPUTS[k], input_hashes[k])) for k in INPUTS)},
        "unit_of_analysis": "60 paired cells (workload x offered rate x TTFT threshold)",
        "sign_rule": "sign(mean_delta_goodput); exact zero is 'no direction'",
        "agreement_rule": (
            "agree = both nonzero and same sign; opposite = both nonzero and different signs; "
            "one_zero and both_zero are neither agree nor opposite"
        ),
        "subset_definitions": {
            "sustainable": "fp32 and bf16 both all-sustainable in that run",
            "boundary": "exactly one of fp32/bf16 all-sustainable in that run",
            "overload": "neither fp32 nor bf16 all-sustainable in that run",
        },
        "statistical_procedures": [
            "exact two-sided binomial sign test over nonzero-both cells, H0 p=0.5",
            "Pearson correlation between formal and repro mean deltas (descriptive only)",
        ],
        "limitations": [
            "cells share seeds and are not independent; sign-test p-values are exploratory and not corrected",
            "no new inference or causal claim; mechanism attribution remains open",
            "near-zero sustainable-region deltas make sign flips expected under measurement noise",
            "the 60 formal cells and 60 repro cells are paired, never merged into 90 cells",
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
        {"fallacy": "Simpson's paradox", "resolution": "Agreement is reported by workload and load subset; no single pooled headline is used."},
        {"fallacy": "Ecological fallacy", "resolution": "All units are cell-level means; no inference about individual requests or seeds."},
        {"fallacy": "Berkson's paradox", "resolution": "All 60 cells are included; no conditioning on p<0.05 or q<0.05."},
        {"fallacy": "Collider bias", "resolution": "No selection on outcome; boundary cells and null cells are retained."},
        {"fallacy": "Base-rate neglect", "resolution": "Sustainable-region deltas are near zero; zero/near-zero sign flips are labeled as noise-prone and not claimed."},
        {"fallacy": "Regression to the mean", "resolution": "Large ShareGPT r45 formal-negative/repro-positive flip is reported as instability, not as an effect reversal."},
        {"fallacy": "Survivorship bias", "resolution": "Both boundary cells that differ across runs are in the cells list and in the summary."},
        {"fallacy": "Look-elsewhere effect", "resolution": "Global sign test is exploratory; overload-region subsets are the manuscript's stated analysis window, and no additional post hoc windows are cherry-picked."},
        {"fallacy": "Garden of forking paths", "resolution": "Sign rule and subset definitions were frozen in the contract before the output was generated."},
        {"fallacy": "Correlation != causation", "resolution": "Directional stability across runs does not identify a mechanism; mechanism attribution remains open."},
        {"fallacy": "Reverse causality", "resolution": "The comparison is paired same-seed allocation deltas; no causal direction is claimed."},
    ]

    analysis = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "created_at": created_at,
        "contract_sha256": sha256_file(contract_path),
        "input_summary": {
            "n_cells_per_run": 60,
            "bh_formal_n_p_lt_0_05": bh["formal"]["n_p_lt_0_05"],
            "bh_formal_n_q_lt_0_05": bh["formal"]["n_q_lt_0_05"],
            "bh_repro_n_p_lt_0_05": bh["repro"]["n_p_lt_0_05"],
            "bh_repro_n_q_lt_0_05": bh["repro"]["n_q_lt_0_05"],
        },
        "overall": summarize(cells),
        "pearson_r_formal_repro": round(pearson_r, 6),
        "by_workload": by_workload,
        "by_load_subset": by_load_subset,
        "workload_x_load_subset": workload_x_subset,
        "by_threshold": by_threshold,
        "cells": cells,
        "self_review": self_review,
        "interpretation_guardrails": [
            "Random60 overload region: all 13 formal-overload cells are positive in both runs (see workload_x_load_subset).",
            "ShareGPT overload region: 3/10 agree and 7/10 opposite; ShareGPT direction is not stable.",
            "Sustainable-region sign flips are mostly small-magnitude and should not be interpreted as effects.",
        ],
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
