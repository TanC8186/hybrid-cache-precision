"""Validate and aggregate a steady-state serving attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

VALID_STATUS = "completed_validated"


class AnalysisError(RuntimeError):
    """Raised when an attempt cannot be analyzed without hiding missing data."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AnalysisError(f"expected JSON object: {path}")
    return value


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def t_critical_95(n: int) -> float:
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    return table.get(n, 1.96)


def summarize(values: Sequence[float]) -> dict[str, Any]:
    parsed = [float(value) for value in values]
    if not parsed:
        return {"n": 0, "mean": None, "std": None, "ci95_half_width": None}
    mean = statistics.fmean(parsed)
    if len(parsed) == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95_half_width": None}
    std = statistics.stdev(parsed)
    ci = t_critical_95(len(parsed)) * std / math.sqrt(len(parsed))
    return {"n": len(parsed), "mean": mean, "std": std, "ci95_half_width": ci}


def verify_sample(sample_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    status = load_json(sample_dir / "status.json")
    if status.get("status") != VALID_STATUS:
        raise AnalysisError(f"sample is not validated: {sample_dir.name} status={status.get('status')!r}")
    result_path = sample_dir / "result.json"
    analysis_path = sample_dir / "analysis.json"
    if sha256_file(result_path) != status.get("result_sha256"):
        raise AnalysisError(f"result hash mismatch: {sample_dir.name}")
    if sha256_file(analysis_path) != status.get("analysis_sha256"):
        raise AnalysisError(f"analysis hash mismatch: {sample_dir.name}")
    return load_json(result_path), load_json(analysis_path)


def selected_thresholds(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row["analysis"]["slo_sweep"])
    return sorted(keys, key=float)


def aggregate_attempt(
    attempt_dir: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    contract = load_json(attempt_dir / "attempt_contract.json")
    expected_plan = contract["plan"]
    expected_ids = {sample["sample_id"] for sample in expected_plan}
    sample_root = attempt_dir / "samples"
    observed_ids = {path.name for path in sample_root.iterdir() if path.is_dir()} if sample_root.exists() else set()
    missing_ids = sorted(expected_ids - observed_ids)
    extra_ids = sorted(observed_ids - expected_ids)
    if extra_ids:
        raise AnalysisError(f"unexpected sample directories: {extra_ids}")
    if missing_ids and not allow_partial:
        raise AnalysisError(f"missing samples: {missing_ids}")

    rows: list[dict[str, Any]] = []
    for sample in expected_plan:
        sample_id = sample["sample_id"]
        if sample_id in missing_ids:
            continue
        result, analysis = verify_sample(sample_root / sample_id)
        rows.append({"sample": sample, "result": result, "analysis": analysis})

    cell_rows: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample = row["sample"]
        key = (
            sample["allocation"],
            sample["workload"],
            float(sample["request_rate"]),
        )
        cell_rows[key].append(row)

    thresholds = selected_thresholds(rows)
    cells: list[dict[str, Any]] = []
    for (allocation, workload, rate), grouped in sorted(cell_rows.items()):
        grouped.sort(key=lambda row: int(row["sample"]["seed"]))
        seeds = [int(row["sample"]["seed"]) for row in grouped]
        metrics: dict[str, Any] = {
            "failed_requests": summarize([row["analysis"]["failed"] for row in grouped]),
            "failed_request_fraction": summarize(
                [row["analysis"]["failed_request_fraction"] for row in grouped]
            ),
            "request_throughput_req_s": summarize([row["analysis"]["request_throughput_req_s"] for row in grouped]),
            "request_throughput_over_offered": summarize(
                [row["analysis"]["request_throughput_over_offered"] for row in grouped]
            ),
            "ttft_p99_ms": summarize([row["analysis"]["ttft_p99_ms_recomputed"] for row in grouped]),
            "tpot_p99_ms": summarize([row["analysis"]["tpot_p99_ms_recomputed"] for row in grouped]),
        }
        for threshold in thresholds:
            metrics[f"goodput_req_s_ttft_{threshold}"] = summarize(
                [row["analysis"]["slo_sweep"][threshold]["goodput_req_s"] for row in grouped]
            )
            metrics[f"goodput_over_offered_ttft_{threshold}"] = summarize(
                [row["analysis"]["slo_sweep"][threshold]["goodput_over_offered"] for row in grouped]
            )
        cells.append(
            {
                "allocation": allocation,
                "workload": workload,
                "request_rate": rate,
                "seeds": seeds,
                "metrics": metrics,
            }
        )

    by_boundary: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample = row["sample"]
        for threshold, sweep in row["analysis"]["slo_sweep"].items():
            by_boundary[
                (
                    sample["allocation"],
                    sample["workload"],
                    int(sample["seed"]),
                    threshold,
                )
            ].append(
                {
                    "rate": float(sample["request_rate"]),
                    "sustainable": bool(sweep["sustainable"]),
                }
            )

    seed_boundaries: list[dict[str, Any]] = []
    for (allocation, workload, seed, threshold), points in sorted(by_boundary.items()):
        points.sort(key=lambda point: point["rate"])
        sustainable_rates = [point["rate"] for point in points if point["sustainable"]]
        boundary = max(sustainable_rates) if sustainable_rates else None
        seed_boundaries.append(
            {
                "allocation": allocation,
                "workload": workload,
                "seed": seed,
                "ttft_threshold_ms": float(threshold),
                "max_tested_sustainable_rate": boundary,
                "no_tested_rate_sustainable": boundary is None,
                "right_censored": (boundary is not None and boundary == points[-1]["rate"]),
                "tested_rates": [point["rate"] for point in points],
            }
        )

    boundary_groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for boundary in seed_boundaries:
        boundary_groups[
            (
                boundary["allocation"],
                boundary["workload"],
                boundary["ttft_threshold_ms"],
            )
        ].append(boundary)

    boundaries: list[dict[str, Any]] = []
    for (allocation, workload, threshold), grouped in sorted(boundary_groups.items()):
        values = [
            boundary["max_tested_sustainable_rate"]
            for boundary in grouped
            if boundary["max_tested_sustainable_rate"] is not None
        ]
        boundaries.append(
            {
                "allocation": allocation,
                "workload": workload,
                "ttft_threshold_ms": threshold,
                "summary": summarize(values),
                "seeds_total": len(grouped),
                "seeds_with_boundary": len(values),
                "any_no_sustainable_rate": any(boundary["no_tested_rate_sustainable"] for boundary in grouped),
                "any_right_censored": any(boundary["right_censored"] for boundary in grouped),
                "per_seed": grouped,
            }
        )

    paired: list[dict[str, Any]] = []
    row_index = {
        (
            row["sample"]["allocation"],
            row["sample"]["workload"],
            float(row["sample"]["request_rate"]),
            int(row["sample"]["seed"]),
        ): row
        for row in rows
    }
    workloads = sorted({row["sample"]["workload"] for row in rows})
    rates = sorted({float(row["sample"]["request_rate"]) for row in rows})
    seeds = sorted({int(row["sample"]["seed"]) for row in rows})
    for workload in workloads:
        for rate in rates:
            for threshold in thresholds:
                differences: list[float] = []
                paired_seeds: list[int] = []
                for seed in seeds:
                    fp16 = row_index.get(("fp16", workload, rate, seed))
                    int4 = row_index.get(("int4", workload, rate, seed))
                    if fp16 is None or int4 is None:
                        continue
                    fp16_goodput = fp16["analysis"]["slo_sweep"][threshold]["goodput_req_s"]
                    int4_goodput = int4["analysis"]["slo_sweep"][threshold]["goodput_req_s"]
                    differences.append(int4_goodput - fp16_goodput)
                    paired_seeds.append(seed)
                if differences:
                    paired.append(
                        {
                            "workload": workload,
                            "request_rate": rate,
                            "ttft_threshold_ms": float(threshold),
                            "metric": "int4_minus_fp16_goodput_req_s",
                            "seeds": paired_seeds,
                            "summary": summarize(differences),
                        }
                    )

    complete = not missing_ids
    return {
        "schema_version": 1,
        "source_attempt": str(attempt_dir.resolve()),
        "attempt_id": contract["attempt_id"],
        "phase": contract["phase"]["name"],
        "verification_status": "ANALYZED" if complete else "QUARANTINED",
        "complete_denominator": complete,
        "expected_samples": len(expected_ids),
        "observed_samples": len(rows),
        "missing_sample_ids": missing_ids,
        "thresholds_ms": [float(value) for value in thresholds],
        "cells": cells,
        "boundaries": boundaries,
        "paired_differences": paired,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Verification Status: {report['verification_status']}",
        "- Version Label: steady_state_validation_v1",
        "",
        "## Steady-State Validation",
        "",
        f"- Attempt: `{report['attempt_id']}`",
        f"- Phase: `{report['phase']}`",
        (f"- Denominator: {report['observed_samples']} / {report['expected_samples']} samples"),
        "",
        "### Sustainable Offered-Rate Boundary",
        "",
        "| Allocation | Workload | TTFT threshold | Seeds | Mean rate | Std | 95% CI half-width | Flags |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for boundary in report["boundaries"]:
        summary = boundary["summary"]
        flags: list[str] = []
        if boundary["any_no_sustainable_rate"]:
            flags.append("below tested range")
        if boundary["any_right_censored"]:
            flags.append("at upper tested bound")
        mean = "NA" if summary["mean"] is None else f"{summary['mean']:.2f}"
        std = "NA" if summary["std"] is None else f"{summary['std']:.2f}"
        ci = "NA" if summary["ci95_half_width"] is None else f"{summary['ci95_half_width']:.2f}"
        lines.append(
            "| {allocation} | {workload} | {threshold:.0f} ms | "
            "{seeds}/{total} | {mean} | {std} | {ci} | {flags} |".format(
                allocation=boundary["allocation"],
                workload=boundary["workload"],
                threshold=boundary["ttft_threshold_ms"],
                seeds=boundary["seeds_with_boundary"],
                total=boundary["seeds_total"],
                mean=mean,
                std=std,
                ci=ci,
                flags=", ".join(flags) or "none",
            )
        )
    lines.extend(
        [
            "",
            "### Integrity",
            "",
            ("- No sample is included unless its status is `completed_validated` and both result hashes match."),
            (
                "- Accounted request failures remain in the offered denominator and count as SLO misses; "
                "they are never treated as zero-latency successes."
            ),
            ("- Boundary summaries retain below-range and right-censored seeds instead of silently dropping them."),
            "- Quantitative paper promotion still requires a reproducibility run.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    attempt_dir = args.attempt_dir.resolve()
    report = aggregate_attempt(attempt_dir, allow_partial=args.allow_partial)
    output_json = args.output_json or attempt_dir / "aggregate.json"
    output_md = args.output_md or attempt_dir / "validation.md"
    atomic_write(
        output_json.resolve(),
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    atomic_write(output_md.resolve(), render_markdown(report))
    print(
        json.dumps(
            {
                "attempt_id": report["attempt_id"],
                "verification_status": report["verification_status"],
                "samples": (f"{report['observed_samples']}/{report['expected_samples']}"),
                "output_json": str(output_json.resolve()),
                "output_md": str(output_md.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
