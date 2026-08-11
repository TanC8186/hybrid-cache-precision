#!/usr/bin/env python3
"""Analyze the RULER no-think 5-cell rerun and compare it with think-default.

Pure analysis over atomic no-think cells. Every cell must be
completed_validated, thinking=disabled, max_tokens=256. Old no-think
attempts and think-default cells are never pooled into the no-think
denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


CELLS = [
    ("ruler_fwe", 4096, "2b"),
    ("ruler_fwe", 8192, "2b"),
    ("ruler_niah_multiquery", 4096, "9b"),
    ("ruler_niah_multiquery", 8192, "9b"),
    ("ruler_fwe", 8192, "9b"),
]
DATASET_SEEDS = [42, 11, 23]


def t_half(n: int, sd: float) -> float:
    df = n - 1
    table = {1: 12.706, 2: 4.303}
    return table.get(df, 1.96) * sd / math.sqrt(n)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def load_cell(ruler_dir: Path, attempt: str, task: str, length: int, dseed: int, alloc: str) -> dict:
    if dseed == 42:
        name = f"{task}__L{length}__{alloc}__s7.json"
    else:
        name = f"{task}__L{length}__{alloc}__s7__d{dseed}.json"
    path = ruler_dir / attempt / name
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("status") != "completed_validated":
        raise SystemExit(f"cell not completed: {path}")
    if rec.get("thinking") != "disabled":
        raise SystemExit(f"cell not no-think: {path}")
    if rec.get("max_tokens") != 256:
        raise SystemExit(f"cell max_tokens != 256: {path}")
    return rec


def interval_overlap(a: list[float], b: list[float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler-dir", default="results/quality/ruler-subset")
    ap.add_argument("--attempt-2b", default="ruler-statebf16-nothink-20260811-2b")
    ap.add_argument("--attempt-9b", default="ruler-statebf16-nothink-20260811-9b")
    ap.add_argument(
        "--think-analysis",
        default="results/quality/ruler-statebf16-multiseed-analysis-20260809.json",
    )
    ap.add_argument("--out", default="results/quality/ruler-nothink-5cell-analysis-20260811.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    ruler_dir = root / args.ruler_dir
    think = json.loads((root / args.think_analysis).read_text(encoding="utf-8"))
    think_map = {(r["task"], r["length"], r["model"]): r for r in think["rows"]}

    input_paths = {
        "think_analysis": root / args.think_analysis,
        "runner_script": root / "scripts" / "eval" / "run_ruler_statebf16_nothink_5cell.sh",
        "plan_doc": root / "docs" / "notes" / "ruler-nothink-5cell-plan-2026-08-11.md",
    }
    input_hashes = {key: sha256_file(path) for key, path in input_paths.items()}
    contract_path = root / "results" / "quality" / "ruler-nothink-5cell-analysis-20260811.contract.json"
    contract = {
        "schema_version": 1,
        "analysis_id": "ruler-nothink-5cell-20260811",
        "created_at": "2026-08-11T00:00:00+08:00",
        "status": "FROZEN",
        "git_commit": git_head(),
        "purpose": "Analyze RULER no-think 5-cell rerun and compare with think-default deltas.",
        "cells": CELLS,
        "dataset_seeds": DATASET_SEEDS,
        "engine_seed": 7,
        "protocol": "ruler_quality.py --max-tokens 256 --disable-thinking --resume",
        "decision_rules": [
            "only completed_validated, thinking=disabled, max_tokens=256 cells enter the analysis",
            "old no-think attempts and think-default cells are never pooled into the no-think denominator",
            "per-cell paired delta with 95% t-CI over 3 dataset seeds",
            "if no-think sign/magnitude changes materially, paper wording must be revised",
        ],
        "inputs": {key: {"path": str(path), "sha256": h} for key, (path, h) in
                   ((k, (input_paths[k], input_hashes[k])) for k in input_paths)},
        "outputs": {
            "analysis_json": str(root / args.out),
            "contract_json": str(contract_path),
        },
    }
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (Path(str(contract_path) + ".sha256")).write_text(sha256_file(contract_path) + "\n", encoding="ascii")

    rows = []
    for task, length, model in CELLS:
        attempt = args.attempt_2b if model == "2b" else args.attempt_9b
        fp16_vals = []
        bf16_vals = []
        per_seed = {}
        for dseed in DATASET_SEEDS:
            fp16 = load_cell(ruler_dir, attempt, task, length, dseed, "fp16")
            bf16 = load_cell(ruler_dir, attempt, task, length, dseed, "fp16_statebf16")
            fp16_vals.append(float(fp16["accuracy"]))
            bf16_vals.append(float(bf16["accuracy"]))
            per_seed[str(dseed)] = {
                "fp16": float(fp16["accuracy"]),
                "fp16_statebf16": float(bf16["accuracy"]),
            }
        diffs = [b - f for f, b in zip(fp16_vals, bf16_vals)]
        mean_d = statistics.mean(diffs)
        sd_d = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        half = t_half(len(diffs), sd_d)
        ci = [mean_d - half, mean_d + half]
        think_row = think_map[(task, length, model)]
        think_ci = think_row["ci95_delta"]
        rows.append(
            {
                "task": task,
                "length": length,
                "model": model,
                "fp16_mean": round(statistics.mean(fp16_vals), 2),
                "bf16_mean": round(statistics.mean(bf16_vals), 2),
                "nothink_delta_mean": round(mean_d, 2),
                "nothink_ci95_delta": [round(ci[0], 2), round(ci[1], 2)],
                "think_delta_mean": think_row["delta_mean"],
                "think_ci95_delta": think_ci,
                "sign_change_think_vs_nothink": (
                    (mean_d > 0) != (think_row["delta_mean"] > 0)
                ),
                "ci_overlap_think_vs_nothink": interval_overlap(ci, think_ci),
                "abs_delta_change": round(abs(mean_d) - abs(think_row["delta_mean"]), 2),
                "per_dataset_seed": per_seed,
            }
        )

    result = {
        "schema_version": 1,
        "analysis_id": "ruler-nothink-5cell-20260811",
        "contract_sha256": sha256_file(contract_path),
        "dataset_seeds": DATASET_SEEDS,
        "engine_seed": 7,
        "protocol": "ruler_quality.py --max-tokens 256 --disable-thinking; dataset generated per random_seed",
        "attempts": {"2b": args.attempt_2b, "9b": args.attempt_9b},
        "n_cells": len(rows),
        "rows": rows,
        "self_review": [
            {"fallacy": "Look-elsewhere", "resolution": "5 cells are fixed before analysis; no new cells are selected."},
            {"fallacy": "Survivorship", "resolution": "Missing/failed cells cause fail-closed exit; they are not dropped."},
            {"fallacy": "Garden of forking paths", "resolution": "Protocol, seeds, and comparison rules are frozen in the plan document."},
            {"fallacy": "Regression to the mean", "resolution": "No-think is a protocol correction; think data is reported as sensitivity, not as the target."},
            {"fallacy": "Correlation != causation", "resolution": "Delta direction change is reported, not attributed to a mechanism."},
        ],
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (Path(str(out) + ".sha256")).write_text(sha256_file(out) + "\n", encoding="ascii")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
