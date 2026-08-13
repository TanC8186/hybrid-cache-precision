"""Logical and statistical audit for the M4 four-allocation serving formal run.

The inferential unit is one seeded allocation/workload/rate cell. Request rows
are used only to produce the cell-level serving metrics and are never treated
as independent repeats. This tool deliberately performs no script hash gate;
the user requested logic review rather than script digest verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

ALLOCATIONS = ("full", "kv_only", "state_only", "joint")
ALTERNATIVES = ("kv_only", "state_only", "joint")
PRECISION_PROFILES = {
    "full": ("auto", "float32"),
    "kv_only": ("int4_per_token_head", "float32"),
    "state_only": ("auto", "bfloat16"),
    "joint": ("int4_per_token_head", "bfloat16"),
}
WORKLOAD_RATES = {
    "random": (30.0, 35.0, 40.0, 45.0, 50.0),
    "sharegpt": (20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0),
}
SEEDS = (11, 23, 47)
TTFT_THRESHOLDS = (250.0, 500.0, 1000.0, 2000.0, 3000.0)
TPOT_THRESHOLD = 200.0
T_CRITICAL_DF2 = float(student_t.ppf(0.975, 2))
EXPECTED_SAMPLES = 144
EXPECTED_REQUESTS = 320_400
EPSILON = 1e-12


class AuditError(RuntimeError):
    """Raised when a required logical integrity condition fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def finite(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{name} is not numeric: {value!r}") from exc
    require(math.isfinite(parsed), f"{name} is not finite: {parsed!r}")
    return parsed


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuditError(f"{name} is not an ISO-8601 timestamp: {value!r}") from exc
    require(parsed.tzinfo is not None, f"{name} has no timezone")
    return parsed


def command_value(command: list[str], flag: str, name: str) -> str:
    require(command.count(flag) == 1, f"{name}: expected exactly one {flag}")
    index = command.index(flag)
    require(index + 1 < len(command), f"{name}: missing value after {flag}")
    return command[index + 1]


def audit_server_sessions(attempt_dir: Path) -> list[dict[str, Any]]:
    server_root = attempt_dir / "servers"
    require(server_root.is_dir(), "server session root is missing")
    observed_allocations = {path.name for path in server_root.iterdir() if path.is_dir()}
    require(observed_allocations == set(ALLOCATIONS), "server allocation directories drift")
    sessions = []
    for allocation in ALLOCATIONS:
        candidates = sorted(path for path in (server_root / allocation).iterdir() if path.is_dir())
        require(len(candidates) == 1, f"{allocation}: expected exactly one server session")
        session_dir = candidates[0]
        contract = load_json(session_dir / "contract.json")
        status = load_json(session_dir / "status.json")
        require(contract.get("allocation") == allocation, f"{allocation}: server contract allocation drift")
        command = contract.get("command")
        require(isinstance(command, list) and all(isinstance(item, str) for item in command), f"{allocation}: invalid server command")
        expected_kv, expected_state = PRECISION_PROFILES[allocation]
        require(command_value(command, "--kv-cache-dtype", allocation) == expected_kv, f"{allocation}: KV precision drift")
        require(
            command_value(command, "--mamba-ssm-cache-dtype", allocation) == expected_state,
            f"{allocation}: state precision drift",
        )
        require(status.get("status") == "stopped", f"{allocation}: server session did not stop cleanly")
        require(int(status.get("returncode", -1)) == 0, f"{allocation}: server return code is nonzero")
        require(status.get("exception") is None, f"{allocation}: server session recorded an exception")
        started = parse_timestamp(str(contract.get("started_at", "")), f"{allocation}.started_at")
        updated = parse_timestamp(str(status.get("updated_at", "")), f"{allocation}.updated_at")
        require(updated >= started, f"{allocation}: server timestamps are not monotonic")
        server_log = session_dir / "server.log"
        require(server_log.is_file() and server_log.stat().st_size > 0, f"{allocation}: server log is missing or empty")
        log_text = server_log.read_text(encoding="utf-8", errors="replace")
        required_log_substrings = (
            f"'mamba_ssm_cache_dtype': '{expected_state}'",
            f"kv_cache_dtype={expected_kv}",
            "CUDAGraphMode.PIECEWISE",
        )
        for substring in required_log_substrings:
            require(substring in log_text, f"{allocation}: precision log evidence missing {substring!r}")
        sessions.append(
            {
                "allocation": allocation,
                "session_id": session_dir.name,
                "kv_cache_dtype": expected_kv,
                "mamba_ssm_cache_dtype": expected_state,
                "cudagraph_mode": "PIECEWISE",
                "status": status["status"],
                "returncode": int(status["returncode"]),
                "exception": status["exception"],
                "started_at": contract["started_at"],
                "stopped_at": status["updated_at"],
                "server_log_size_bytes": server_log.stat().st_size,
                "precision_log_evidence": "PASS",
            }
        )
    return sessions


def audit_launch(attempt_dir: Path) -> dict[str, Any]:
    launch_dir = attempt_dir.parent / "launch" / attempt_dir.name
    names = ("pid", "started_at", "finished_at", "exit_code", "run.log")
    files = {name: (launch_dir / name).is_file() for name in names}
    result: dict[str, Any] = {
        "directory": str(launch_dir),
        "files": files,
        "complete": all(files.values()),
    }
    if not result["complete"]:
        missing = [name for name, present in files.items() if not present]
        result["warning"] = "launcher provenance is incomplete; missing: " + ", ".join(missing)
        return result
    try:
        pid = int((launch_dir / "pid").read_text(encoding="ascii").strip())
        exit_code = int((launch_dir / "exit_code").read_text(encoding="ascii").strip())
    except ValueError as exc:
        raise AuditError("launcher pid or exit_code is not an integer") from exc
    require(pid > 0, "launcher PID must be positive")
    require(exit_code == 0, f"launcher exit code is nonzero: {exit_code}")
    started_text = (launch_dir / "started_at").read_text(encoding="ascii").strip()
    finished_text = (launch_dir / "finished_at").read_text(encoding="ascii").strip()
    started = parse_timestamp(started_text, "launcher.started_at")
    finished = parse_timestamp(finished_text, "launcher.finished_at")
    require(finished >= started, "launcher timestamps are not monotonic")
    run_log = launch_dir / "run.log"
    require(run_log.stat().st_size > 0, "launcher run.log is empty")
    result.update(
        {
            "pid": pid,
            "exit_code": exit_code,
            "started_at": started_text,
            "finished_at": finished_text,
            "duration_s": (finished - started).total_seconds(),
            "run_log_size_bytes": run_log.stat().st_size,
            "warning": None,
        }
    )
    return result


def sample_id(allocation: str, workload: str, rate: float, seed: int) -> str:
    return f"{allocation}__{workload}__r{rate:g}__s{seed}"


def expected_plan() -> list[tuple[str, str, float, int]]:
    return [
        (allocation, workload, rate, seed)
        for allocation in ALLOCATIONS
        for workload, rates in WORKLOAD_RATES.items()
        for rate in rates
        for seed in SEEDS
    ]


def metric_from_sample(sample_dir: Path, expected: tuple[str, str, float, int]) -> dict[str, Any]:
    allocation, workload, rate, seed = expected
    sid = sample_id(*expected)
    status = load_json(sample_dir / "status.json")
    contract = load_json(sample_dir / "contract.json")
    analysis = load_json(sample_dir / "analysis.json")
    result = load_json(sample_dir / "result.json")

    require(status.get("status") == "completed_validated", f"{sid}: status is not completed_validated")
    require(int(status.get("returncode", -1)) == 0, f"{sid}: nonzero benchmark return code")
    require(analysis.get("status") == "completed_validated", f"{sid}: analysis status mismatch")
    for key, value in {
        "sample_id": sid,
        "allocation": allocation,
        "workload": workload,
        "request_rate": rate,
        "seed": seed,
    }.items():
        observed = contract.get(key)
        if isinstance(value, float):
            require(math.isclose(float(observed), value, rel_tol=0, abs_tol=1e-9), f"{sid}: contract {key} drift")
        else:
            require(observed == value, f"{sid}: contract {key} drift")
    expected_requests = int(contract["num_prompts"])
    completed = int(result["completed"])
    failed = int(result["failed"])
    require(completed + failed == expected_requests, f"{sid}: request denominator mismatch")
    require(int(analysis["completed"]) == completed, f"{sid}: analysis completed mismatch")
    require(int(analysis["failed"]) == failed, f"{sid}: analysis failed mismatch")
    require(failed == 0, f"{sid}: failed requests are present")
    fields = ("ttfts", "itls", "input_lens", "output_lens", "start_times", "errors")
    for field in fields:
        require(len(result[field]) == expected_requests, f"{sid}: detailed {field} denominator mismatch")

    sweep = analysis.get("slo_sweep")
    require(isinstance(sweep, dict), f"{sid}: missing SLO sweep")
    metrics: dict[str, Any] = {
        "sample_id": sid,
        "allocation": allocation,
        "workload": workload,
        "rate": rate,
        "seed": seed,
        "expected_requests": expected_requests,
        "completed_requests": completed,
        "failed_requests": failed,
        "throughput": finite(f"{sid}.throughput", analysis["request_throughput_req_s"]),
        "ttft_p95": finite(f"{sid}.ttft_p95", analysis["ttft_p95_ms_recomputed"]),
        "ttft_p99": finite(f"{sid}.ttft_p99", analysis["ttft_p99_ms_recomputed"]),
        "tpot_p95": finite(f"{sid}.tpot_p95", analysis["tpot_p95_ms_recomputed"]),
        "tpot_p99": finite(f"{sid}.tpot_p99", analysis["tpot_p99_ms_recomputed"]),
        "arrival_ratio": finite(f"{sid}.arrival_ratio", analysis["arrival_span_over_target"]),
        "duration_s": finite(f"{sid}.duration", analysis["benchmark_duration_s"]),
        "sustainable": {},
    }
    for threshold in TTFT_THRESHOLDS:
        key = f"{threshold:g}"
        require(key in sweep, f"{sid}: missing TTFT threshold {key}")
        row = sweep[key]
        metrics[f"goodput_{key}"] = finite(f"{sid}.goodput_{key}", row["goodput_req_s"])
        metrics["sustainable"][key] = bool(row["sustainable"])
    return metrics


def paired_summary(values: list[float]) -> dict[str, Any]:
    require(len(values) == len(SEEDS), "paired summary requires exactly three seeds")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    margin = T_CRITICAL_DF2 * sd / math.sqrt(len(values))
    # scipy returns NaN for the all-zero constant sample.  Define the
    # degenerate paired test explicitly so BH-FDR receives finite p-values.
    if sd <= EPSILON:
        p_value = 1.0 if abs(mean) <= EPSILON else 0.0
        dz = None
    else:
        test = ttest_1samp(values, 0.0)
        p_value = float(test.pvalue)
        require(math.isfinite(p_value), "paired t-test returned a non-finite p-value")
        dz = mean / sd
    return {
        "n": len(values),
        "values": values,
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "p_value": p_value,
        "cohen_dz": dz,
    }


def bh_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [1.0] * len(p_values)
    running = 1.0
    n = len(p_values)
    for rank in range(n, 0, -1):
        index = order[rank - 1]
        value = min(running, p_values[index] * n / rank)
        running = value
        adjusted[index] = min(1.0, value)
    return adjusted


def effect_class(dz: float | None) -> str:
    if dz is None:
        return "undefined"
    magnitude = abs(dz)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def fallacy_scan(*, launch_complete: bool) -> list[dict[str, str]]:
    provenance_detail = (
        "The frozen formal matrix and complete launcher provenance constrain researcher degrees of freedom."
        if launch_complete
        else "The formal matrix is frozen, but launcher provenance is incomplete; the gap is disclosed rather than repaired retrospectively."
    )
    rows = [
        ("Simpson's paradox", "NOTE", "All primary comparisons stay stratified by workload and offered rate; pooled directions are not used as the sole result."),
        ("Ecological fallacy", "CAUTION", "The inferential unit is the paired seed/trace cell, not an individual request; request counts only form cell metrics."),
        ("Berkson's paradox", "CAUTION", "The evidence is conditional on one model, one RTX 5090, and the frozen load grid; external deployment prevalence is not inferred."),
        ("Collider bias", "NOTE", "No post-treatment queue, latency, or success variable is conditioned on when forming paired differences."),
        ("Base rate neglect", "NOTE", "All 320,400 offered measurement requests remain in denominators and failures are counted as SLO misses."),
        ("Regression to the mean", "NOTE", "Confirmatory seeds 11/23/47 were frozen before this run and were not selected by observed outcomes."),
        ("Survivorship bias", "NOTE", "All 144 cells are retained and the audit fails closed on missing or failed cells; observed failure count is zero."),
        ("Look-elsewhere effect", "CAUTION", "180 primary goodput tests plus secondary metrics and five thresholds are reported; BH-FDR is applied within the declared primary family."),
        ("Garden of forking paths", "CAUTION", provenance_detail),
        ("Correlation != causation", "CAUTION", "Allocation is controlled, but environment-sensitive serving measurements do not by themselves identify a mechanism or universal causal benefit."),
        ("Reverse causality", "NOTE", "Precision allocation is assigned before each cold-start serving epoch, so latency cannot determine the assignment retrospectively."),
    ]
    return [{"fallacy": name, "severity": severity, "detail": detail, "status": "CHECKED"} for name, severity, detail in rows]


def require_no_partial_artifacts(attempt_dir: Path) -> None:
    partials = sorted(attempt_dir.rglob("*.partial"))
    preview = ", ".join(str(path.relative_to(attempt_dir)) for path in partials[:5])
    require(not partials, f"partial artifacts remain: {preview}")


def audit_attempt(attempt_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require_no_partial_artifacts(attempt_dir)
    contract = load_json(attempt_dir / "attempt_contract.json")
    summary = load_json(attempt_dir / "summary.json")
    expected = expected_plan()
    expected_ids = [sample_id(*row) for row in expected]
    contract_plan = contract.get("plan", [])
    require(len(contract_plan) == EXPECTED_SAMPLES, "formal plan is not 144 cells")
    for observed, (allocation, workload, rate, seed) in zip(contract_plan, expected, strict=True):
        sid = sample_id(allocation, workload, rate, seed)
        require(observed.get("sample_id") == sid, f"{sid}: attempt plan sample ID drift")
        require(observed.get("allocation") == allocation, f"{sid}: attempt plan allocation drift")
        require(observed.get("workload") == workload, f"{sid}: attempt plan workload drift")
        require(math.isclose(float(observed.get("request_rate", math.nan)), rate, rel_tol=0, abs_tol=1e-9), f"{sid}: attempt plan rate drift")
        require(int(observed.get("seed", -1)) == seed, f"{sid}: attempt plan seed drift")
        require(int(observed.get("num_prompts", -1)) == round(rate * 60), f"{sid}: attempt plan request denominator drift")
    require(summary.get("counts") == {"completed_validated": EXPECTED_SAMPLES}, "summary counts are incomplete")
    observed_ids = [str(row["sample_id"]) for row in summary.get("samples", [])]
    require(observed_ids == expected_ids, "summary sample order or membership drift")
    rows = []
    for item in expected:
        rows.append(metric_from_sample(attempt_dir / "samples" / sample_id(*item), item))
    require(sum(row["expected_requests"] for row in rows) == EXPECTED_REQUESTS, "expected request denominator drift")
    require(sum(row["completed_requests"] for row in rows) == EXPECTED_REQUESTS, "completed request denominator drift")
    require(sum(row["failed_requests"] for row in rows) == 0, "failed request count is nonzero")
    return contract, rows


def comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["allocation"], r["workload"], r["rate"], r["seed"]): r for r in rows}
    primary: list[dict[str, Any]] = []
    secondary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for workload, rates in WORKLOAD_RATES.items():
        for rate in rates:
            for alternative in ALTERNATIVES:
                for threshold in TTFT_THRESHOLDS:
                    key = f"{threshold:g}"
                    diffs = [
                        by_key[(alternative, workload, rate, seed)][f"goodput_{key}"]
                        - by_key[("full", workload, rate, seed)][f"goodput_{key}"]
                        for seed in SEEDS
                    ]
                    summary = paired_summary(diffs)
                    primary.append({
                        "family": "primary_goodput",
                        "metric": "goodput_req_s",
                        "ttft_threshold_ms": threshold,
                        "allocation": alternative,
                        "baseline": "full",
                        "workload": workload,
                        "rate": rate,
                        "values_by_seed": dict(zip(SEEDS, diffs, strict=True)),
                        **summary,
                    })
                for metric in ("throughput", "ttft_p95", "tpot_p95"):
                    diffs = [
                        by_key[(alternative, workload, rate, seed)][metric]
                        - by_key[("full", workload, rate, seed)][metric]
                        for seed in SEEDS
                    ]
                    summary = paired_summary(diffs)
                    secondary[metric].append({
                        "family": f"secondary_{metric}",
                        "metric": metric,
                        "allocation": alternative,
                        "baseline": "full",
                        "workload": workload,
                        "rate": rate,
                        "values_by_seed": dict(zip(SEEDS, diffs, strict=True)),
                        **summary,
                    })
    primary_q = bh_adjust([float(row["p_value"]) for row in primary])
    for row, q in zip(primary, primary_q, strict=True):
        row["q_value_bh"] = q
        row["bh_significant_q05"] = q < 0.05
    secondary_q = {metric: bh_adjust([float(row["p_value"]) for row in values]) for metric, values in secondary.items()}
    for metric, values in secondary.items():
        for row, q in zip(values, secondary_q[metric], strict=True):
            row["q_value_bh_within_metric"] = q
            row["bh_significant_q05"] = q < 0.05
    return primary + [row for values in secondary.values() for row in values]


def boundaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["allocation"], r["workload"], r["rate"], r["seed"]): r for r in rows}
    out = []
    for allocation in ALLOCATIONS:
        for workload, rates in WORKLOAD_RATES.items():
            for threshold in TTFT_THRESHOLDS:
                key = f"{threshold:g}"
                sustainable_rates = [
                    rate for rate in rates
                    if all(by_key[(allocation, workload, rate, seed)]["sustainable"][key] for seed in SEEDS)
                ]
                out.append({
                    "allocation": allocation,
                    "workload": workload,
                    "ttft_threshold_ms": threshold,
                    "all_seed_sustainable_rates": sustainable_rates,
                    "boundary_req_s": max(sustainable_rates) if sustainable_rates else None,
                })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    attempt_dir = args.attempt_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    contract, rows = audit_attempt(attempt_dir)
    server_sessions = audit_server_sessions(attempt_dir)
    all_comparisons = comparisons(rows)
    primary = [row for row in all_comparisons if row["family"] == "primary_goodput"]
    secondary = [row for row in all_comparisons if row["family"] != "primary_goodput"]
    launch = audit_launch(attempt_dir)
    integrity = {
        "gate3_status": "PASS",
        "planned_cells": EXPECTED_SAMPLES,
        "completed_cells": len(rows),
        "failed_cells": 0,
        "planned_measurement_requests": EXPECTED_REQUESTS,
        "completed_measurement_requests": sum(row["completed_requests"] for row in rows),
        "failed_measurement_requests": sum(row["failed_requests"] for row in rows),
        "silent_exclusions": 0,
        "partial_artifacts": len(list(attempt_dir.rglob("*.partial"))),
        "server_sessions": len(server_sessions),
        "server_session_audit": server_sessions,
        "launch_provenance": launch,
        "launch_provenance_files": launch["files"],
        "launch_provenance_warning": launch["warning"],
        "script_hash_gate": "SKIPPED_PER_USER_REQUEST_LOGIC_REVIEW_ONLY",
    }
    report = {
        "schema_version": 1,
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "origin_date": now_utc()[:10],
            "verification_status": "ANALYZED",
            "version_label": "m4_four_config_formal_validation_v1",
        },
        "generated_at": now_utc(),
        "attempt_id": contract["attempt_id"],
        "parent_attempt": contract.get("parent_attempt"),
        "root_commit": contract.get("git_commit"),
        "vllm_source_commit": contract.get("vllm_source_commit"),
        "determinism_class": "environment_sensitive_seeded_serving_benchmark",
        "evidence_status": "ANALYZED",
        "overall_confidence": "CAUTION",
        "integrity": integrity,
        "statistical_unit": "allocation x workload x offered_rate x seed; requests are not independent repeats",
        "statistical_method": {
            "primary_endpoint": "paired goodput_req_s at each predeclared TTFT threshold with TPOT <= 200 ms",
            "baseline": "full",
            "alternatives": list(ALTERNATIVES),
            "seeds": list(SEEDS),
            "ci": "two-sided 95% Student t interval on three paired seed differences (df=2)",
            "t_critical_df2": T_CRITICAL_DF2,
            "effect_size": "Cohen dz = mean paired difference / paired SD",
            "multiple_comparisons": "Benjamini-Hochberg within the 180-test primary goodput family; secondary metric families corrected separately",
            "timing_metrics": "reported descriptively; no wall-clock reproducibility comparison",
        },
        "primary_comparisons": primary,
        "secondary_comparisons": secondary,
        "sustainable_boundaries": boundaries(rows),
        "fallacy_scan_coverage": "11/11",
        "fallacy_scan": fallacy_scan(launch_complete=bool(launch["complete"])),
        "reproducibility": {
            "method": "pending new-attempt environment-sensitive re-run",
            "verdict": "CANNOT_VERIFY",
            "reason": "This attempt is structurally complete, but Gate 4 requires comparison against a separate environment-sensitive formal attempt.",
        },
        "promotion": {
            "paper_quantitative_use_authorized": False,
            "next_gate": "separate-attempt environment-sensitive run-stability comparison",
        },
    }
    (out_dir / "m4_formal_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "m4_formal_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["family", "metric", "ttft_threshold_ms", "allocation", "baseline", "workload", "rate", "n", "mean", "sd", "ci95_low", "ci95_high", "p_value", "q_value_bh", "q_value_bh_within_metric", "cohen_dz", "bh_significant_q05"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_comparisons)
    significant = [row for row in primary if row.get("bh_significant_q05")]
    launch_sentence = (
        "Launcher provenance is complete and records exit code 0."
        if launch["complete"]
        else "The detached launch has incomplete provenance; the gap is retained as a warning and is not backfilled."
    )
    primary_family_sentence = (
        f"Primary family: {len(primary)} paired goodput comparisons "
        "(3 alternatives x [5 Random + 7 ShareGPT rates] x 5 TTFT thresholds), "
        f"with seed/trace as the repeat unit. BH-FDR q<0.05 survivors: {len(significant)}."
    )
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {report['material_passport']['origin_date']}",
        "- Verification Status: ANALYZED",
        "- Version Label: m4_four_config_formal_validation_v1",
        "",
        "## Validation Report",
        "",
        f"- Source: `{contract['attempt_id']}`",
        "- Overall Confidence: CAUTION",
        "- Evidence Status: ANALYZED; not promoted before Gate 4",
        "",
        "### Integrity",
        "",
        f"The frozen formal matrix passed logical Gate 3: {len(rows)}/144 cells, {integrity['completed_measurement_requests']}/{EXPECTED_REQUESTS} measurement requests, zero failed requests, zero partial artifacts, and four precision-verified stopped server sessions. {launch_sentence}",
        "",
        "### Statistical Findings",
        "",
        primary_family_sentence,
        "",
        "| Metric | Comparison | Workload | Rate | Mean delta | 95% CI | p | BH q | dz |",
        "|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in sorted(primary, key=lambda r: (r["workload"], r["rate"], r["allocation"], r["ttft_threshold_ms"]))[:24]:
        lines.append(f"| goodput @ {row['ttft_threshold_ms']:g}ms | {row['allocation']} - full | {row['workload']} | {row['rate']:g} | {row['mean']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['p_value']:.4g} | {row['q_value_bh']:.4g} | {row['cohen_dz'] if row['cohen_dz'] is not None else 'NA'} |")
    lines += [
        "",
        "Only the first 24 primary rows are shown in this compact report; the complete comparison table is in `m4_formal_comparisons.csv`.",
        "",
        "### Warnings",
        "",
        "- n=3 paired repeats gives wide, assumption-sensitive CIs; request-level rows are not independent.",
        "- Serving is environment-sensitive; latency and throughput should not be treated as exact deterministic quantities.",
        (
            "- Launcher provenance is complete; this individual attempt remains ANALYZED until the separate-attempt comparison is evaluated."
            if launch["complete"]
            else "- Launcher provenance is incomplete, so this attempt remains ANALYZED pending a separate attempt with complete provenance."
        ),
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
    lines += [
        "",
        "### Reproducibility",
        "",
        "- Method: pending separate-attempt environment-sensitive run-stability comparison",
        "- Verdict: CANNOT_VERIFY",
        "- Promotion: quantitative paper use is blocked until the new attempt passes structural and tolerance comparison.",
        "",
    ]
    (out_dir / "m4_formal_validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ANALYZED", "cells": len(rows), "requests": integrity["completed_measurement_requests"], "primary_tests": len(primary), "bh_survivors": len(significant), "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        raise SystemExit(f"AUDIT_ERROR: {exc}")
