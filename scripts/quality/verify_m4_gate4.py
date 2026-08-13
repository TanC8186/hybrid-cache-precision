"""Compare the two M4 formal attempts as environment-sensitive run stability.

The same frozen seeds are used in both attempts, so this is a second formal run
and not an independent statistical sample.  Cell-level SLO goodput is compared
with a predeclared 10% symmetric relative tolerance; timing metrics are not used
as reproducibility gates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TOLERANCE = 0.10
EPSILON = 1e-12


class VerificationError(RuntimeError):
    """Raised when a Gate 4 integrity or stability condition fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_analyzer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("analyze_m4_formal", path)
    require(spec is not None and spec.loader is not None, f"cannot load analyzer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def symmetric_relative_difference(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), EPSILON)
    return abs(left - right) / denominator


def sample_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped = {str(row["sample_id"]): row for row in rows}
    require(len(mapped) == len(rows), "duplicate sample IDs in audited rows")
    return mapped


def compare_cells(
    original_rows: list[dict[str, Any]],
    rerun_rows: list[dict[str, Any]],
    *,
    thresholds: tuple[float, ...],
    tolerance: float,
) -> list[dict[str, Any]]:
    original = sample_map(original_rows)
    rerun = sample_map(rerun_rows)
    require(original.keys() == rerun.keys(), "attempt sample memberships differ")
    comparisons = []
    for sample_id in sorted(original):
        left = original[sample_id]
        right = rerun[sample_id]
        require(left["expected_requests"] == right["expected_requests"], f"{sample_id}: expected denominator drift")
        require(left["completed_requests"] == right["completed_requests"], f"{sample_id}: completed denominator drift")
        require(left["failed_requests"] == right["failed_requests"] == 0, f"{sample_id}: failed request drift")
        for threshold in thresholds:
            key = f"{threshold:g}"
            original_value = float(left[f"goodput_{key}"])
            rerun_value = float(right[f"goodput_{key}"])
            difference = symmetric_relative_difference(original_value, rerun_value)
            comparisons.append(
                {
                    "sample_id": sample_id,
                    "allocation": left["allocation"],
                    "workload": left["workload"],
                    "rate": left["rate"],
                    "seed": left["seed"],
                    "ttft_threshold_ms": threshold,
                    "original_goodput_req_s": original_value,
                    "rerun_goodput_req_s": rerun_value,
                    "symmetric_relative_difference": difference,
                    "within_tolerance": difference < tolerance,
                    "original_sustainable": bool(left["sustainable"][key]),
                    "rerun_sustainable": bool(right["sustainable"][key]),
                    "boundary_point_exact": bool(left["sustainable"][key]) == bool(right["sustainable"][key]),
                }
            )
    return comparisons


def boundary_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float], float | None]:
    return {
        (str(row["allocation"]), str(row["workload"]), float(row["ttft_threshold_ms"])): row["boundary_req_s"]
        for row in rows
    }


def compare_boundaries(original: list[dict[str, Any]], rerun: list[dict[str, Any]]) -> list[dict[str, Any]]:
    left = boundary_map(original)
    right = boundary_map(rerun)
    require(left.keys() == right.keys(), "boundary memberships differ")
    return [
        {
            "allocation": key[0],
            "workload": key[1],
            "ttft_threshold_ms": key[2],
            "original_boundary_req_s": left[key],
            "rerun_boundary_req_s": right[key],
            "exact_match": left[key] == right[key],
        }
        for key in sorted(left)
    ]


def fallacy_scan() -> list[dict[str, str]]:
    rows = [
        ("Simpson's paradox", "NOTE", "Run stability is compared per allocation, workload, rate, seed, and threshold; no pooled direction is substituted."),
        ("Ecological fallacy", "CAUTION", "The comparison unit is a seeded serving cell; requests remain constituents of a cell metric, not independent repeats."),
        ("Berkson's paradox", "NOTE", "All frozen cells are required and no cell is selected by outcome."),
        ("Collider bias", "NOTE", "No post-treatment queue or latency variable is conditioned on in the stability comparison."),
        ("Base rate neglect", "NOTE", "Both attempts retain all offered requests and count failures as SLO misses."),
        ("Regression to the mean", "CAUTION", "The rerun uses the same frozen seeds; it is run stability, not a new independent sample."),
        ("Survivorship bias", "NOTE", "Any missing, failed, or incomplete cell blocks verification instead of being omitted."),
        ("Look-elsewhere effect", "NOTE", "The 10% cell-goodput tolerance and complete five-threshold family were frozen before inspecting r3."),
        ("Garden of forking paths", "NOTE", "The rerun command, matrix, tolerance, environment class, and timing-metric exclusion were frozen before completion."),
        ("Correlation != causation", "CAUTION", "Passing run stability supports measurement repeatability in this environment, not a universal mechanism claim."),
        ("Reverse causality", "NOTE", "Allocation is assigned before each cold-start serving session and cannot be selected by observed performance."),
    ]
    return [{"fallacy": name, "severity": severity, "status": "CHECKED", "detail": detail} for name, severity, detail in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--analyzer", type=Path, default=Path(__file__).with_name("analyze_m4_formal.py"))
    args = parser.parse_args()
    require(0.0 < args.tolerance < 1.0, "tolerance must be between 0 and 1")

    analyzer = load_analyzer(args.analyzer.resolve())
    original_dir = args.original.resolve()
    rerun_dir = args.rerun.resolve()
    original_contract, original_rows = analyzer.audit_attempt(original_dir)
    rerun_contract, rerun_rows = analyzer.audit_attempt(rerun_dir)
    original_sessions = analyzer.audit_server_sessions(original_dir)
    rerun_sessions = analyzer.audit_server_sessions(rerun_dir)
    rerun_launch = analyzer.audit_launch(rerun_dir)
    require(rerun_launch["complete"], "rerun launcher provenance is incomplete")
    require(rerun_launch["exit_code"] == 0, "rerun launcher exit code is nonzero")
    require(rerun_contract.get("parent_attempt") == original_contract.get("attempt_id"), "rerun parent linkage mismatch")
    for key in ("git_commit", "vllm_source_commit", "config_sha256", "phase", "plan"):
        require(original_contract.get(key) == rerun_contract.get(key), f"frozen contract field differs: {key}")

    cells = compare_cells(
        original_rows,
        rerun_rows,
        thresholds=analyzer.TTFT_THRESHOLDS,
        tolerance=args.tolerance,
    )
    failures = [row for row in cells if not row["within_tolerance"]]
    boundaries = compare_boundaries(analyzer.boundaries(original_rows), analyzer.boundaries(rerun_rows))
    exact_boundaries = sum(row["exact_match"] for row in boundaries)
    point_exact = sum(row["boundary_point_exact"] for row in cells)
    status = "VERIFIED" if not failures else "ANALYZED"
    verdict = "REPRODUCIBLE" if not failures else "NOT_REPRODUCIBLE"
    confidence = "CAUTION"

    distribution = Counter(
        "<1%" if row["symmetric_relative_difference"] < 0.01 else
        "1-5%" if row["symmetric_relative_difference"] < 0.05 else
        "5-10%" if row["symmetric_relative_difference"] < 0.10 else
        ">=10%"
        for row in cells
    )
    report = {
        "schema_version": 1,
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "origin_date": now_utc()[:10],
            "verification_status": status,
            "version_label": "m4_four_config_gate4_stability_v1",
        },
        "generated_at": now_utc(),
        "original_attempt_id": original_contract["attempt_id"],
        "rerun_attempt_id": rerun_contract["attempt_id"],
        "determinism_class": "environment_sensitive_seeded_serving_benchmark",
        "same_seed_second_formal_run": True,
        "independent_replication": False,
        "verification_status": status,
        "reproducibility_verdict": verdict,
        "overall_confidence": confidence,
        "method": {
            "primary_metric": "cell-level SLO goodput_req_s",
            "cell_comparisons": len(cells),
            "tolerance": "symmetric relative difference < 10%",
            "tolerance_value": args.tolerance,
            "timing_metrics_compared": False,
            "boundary_exactness": "reported descriptively; not a promotion gate because 5 req/s grid points can flip near 0.95",
        },
        "integrity": {
            "original_cells": len(original_rows),
            "rerun_cells": len(rerun_rows),
            "original_requests": sum(row["completed_requests"] for row in original_rows),
            "rerun_requests": sum(row["completed_requests"] for row in rerun_rows),
            "failed_requests": sum(row["failed_requests"] for row in original_rows + rerun_rows),
            "original_server_sessions": original_sessions,
            "rerun_server_sessions": rerun_sessions,
            "rerun_launch": rerun_launch,
            "script_hash_gate": "SKIPPED_PER_USER_REQUEST_LOGIC_REVIEW_ONLY",
        },
        "comparison": {
            "within_tolerance": len(cells) - len(failures),
            "outside_tolerance": len(failures),
            "max_symmetric_relative_difference": max(row["symmetric_relative_difference"] for row in cells),
            "distribution": dict(distribution),
            "boundary_points_exact": point_exact,
            "boundary_points_total": len(cells),
            "boundaries_exact": exact_boundaries,
            "boundaries_total": len(boundaries),
            "cells": cells,
            "boundaries": boundaries,
        },
        "fallacy_scan_coverage": "11/11",
        "fallacy_scan": fallacy_scan(),
        "claim_boundary": [
            "Call this same-seed run stability, not an independent replication or additional n.",
            "Do not pool original and rerun request denominators or seed counts.",
            "Do not compare wall-clock duration as a deterministic reproducibility metric.",
            "Restrict quantitative claims to this model, GPU, software revision, workload, and offered-load grid.",
        ],
    }
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "m4_gate4_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {report['material_passport']['origin_date']}",
        f"- Verification Status: {status}",
        "- Version Label: m4_four_config_gate4_stability_v1",
        "",
        "## Validation Report",
        "",
        f"- Source: `{original_contract['attempt_id']}` and `{rerun_contract['attempt_id']}`",
        f"- Overall Confidence: {confidence}",
        f"- Reproducibility Verdict: {verdict}",
        "- Method: same-seed environment-sensitive second formal run; not independent replication",
        "",
        "### Integrity",
        "",
        f"Both attempts contain {len(original_rows)}/144 validated cells and {report['integrity']['original_requests']}/320400 completed requests with zero failures. The rerun has complete launcher provenance with exit code 0, and all eight server sessions pass realized precision-log checks.",
        "",
        "### Run Stability",
        "",
        f"- Cell-threshold comparisons within the predeclared 10% tolerance: {len(cells) - len(failures)}/{len(cells)}",
        f"- Maximum symmetric relative difference: {100.0 * report['comparison']['max_symmetric_relative_difference']:.3f}%",
        f"- Exact sustainable point labels: {point_exact}/{len(cells)}",
        f"- Exact all-seed boundaries: {exact_boundaries}/{len(boundaries)}",
        f"- Distribution: {dict(distribution)}",
        "",
        "### Fallacy Scan",
        "",
        "- Coverage: 11/11 checked",
        "",
        "| Fallacy | Severity | Detail |",
        "|---|---|---|",
    ]
    for item in report["fallacy_scan"]:
        lines.append(f"| {item['fallacy']} | {item['severity']} | {item['detail']} |")
    lines += ["", "### Claim Boundary", ""] + [f"- {item}" for item in report["claim_boundary"]]
    (out_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verification_status": status,
                "verdict": verdict,
                "cell_comparisons": len(cells),
                "within_tolerance": len(cells) - len(failures),
                "max_relative_difference": report["comparison"]["max_symmetric_relative_difference"],
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        raise SystemExit(f"VERIFICATION_ERROR: {exc}")
