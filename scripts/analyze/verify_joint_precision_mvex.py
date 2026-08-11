"""Audit a four-allocation joint-precision steady-state MVEx attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.analyze.verify_a2_reproduction import (
    VerificationError,
    load_json,
    require,
    sha256_file,
    utc_timestamp,
    verify_sidecar,
    write_json_with_hash,
    write_text_with_hash,
)

EXPECTED_ALLOCATIONS = {
    "full": {"kv_cache_dtype": "auto", "state_dtype": "float32"},
    "kv_only": {"kv_cache_dtype": "int4_per_token_head", "state_dtype": "float32"},
    "state_only": {"kv_cache_dtype": "auto", "state_dtype": "bfloat16"},
    "joint": {"kv_cache_dtype": "int4_per_token_head", "state_dtype": "bfloat16"},
}
FATAL_SERVER_SIGNATURES = (
    "CUDA out of memory",
    "CUDA error: an illegal instruction",
    "CUDA error: an illegal memory access",
    "EngineCore encountered a fatal error",
)
GENERATED_FILES = {
    "artifact_sha256_manifest.json",
    "artifact_sha256_manifest.json.sha256",
    "mvex_audit_report.json",
    "mvex_audit_report.json.sha256",
    "validation_report.md",
    "validation_report.md.sha256",
}


def canonical_sample_contract_sha(contract: Mapping[str, Any]) -> str:
    base = {key: value for key, value in contract.items() if key not in {"command", "contract_sha256"}}
    payload = json.dumps(
        base,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def command_value(command: Sequence[str], flag: str) -> str:
    require(command.count(flag) == 1, f"command must contain exactly one {flag}")
    index = command.index(flag)
    require(index + 1 < len(command), f"command flag has no value: {flag}")
    return str(command[index + 1])


def finite(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{name} is not numeric: {value!r}") from exc
    require(math.isfinite(parsed), f"{name} is not finite: {parsed!r}")
    return parsed


def require_close(name: str, observed: Any, expected: Any, *, tolerance: float = 1e-9) -> None:
    require(
        math.isclose(finite(name, observed), finite(name, expected), rel_tol=tolerance, abs_tol=tolerance),
        f"{name} mismatch: observed={observed!r} expected={expected!r}",
    )


def percentile(values: Sequence[float], pct: float) -> float:
    require(bool(values), "cannot compute percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def per_request_tpot_ms(result: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for output_len, request_itls in zip(result["output_lens"], result["itls"], strict=True):
        count = int(output_len)
        values.append(0.0 if count <= 1 else 1000.0 * sum(map(float, request_itls)) / (count - 1))
    return values


def request_accounting(result: Mapping[str, Any], expected: int) -> dict[str, Any]:
    completed = int(result["completed"])
    failed = int(result["failed"])
    require(completed + failed == expected, "completed + failed does not match the frozen denominator")
    detail_fields = ("ttfts", "itls", "input_lens", "output_lens", "start_times", "errors")
    lengths = {field: len(result[field]) for field in detail_fields}
    require(all(length == expected for length in lengths.values()), f"detailed row mismatch: {lengths}")
    detailed_failures = sum(bool(error) for error in result["errors"])
    require(detailed_failures == failed, "detailed failures do not match result.failed")
    require(expected - detailed_failures == completed, "detailed successes do not match result.completed")
    return {
        "expected": expected,
        "completed": completed,
        "failed": failed,
        "detail_lengths": lengths,
        "request_conservation": True,
    }


def recompute_analysis(
    result: Mapping[str, Any],
    sample: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    accounting = request_accounting(result, int(sample["num_prompts"]))
    success = [not bool(error) for error in result["errors"]]
    ttfts_ms_all = [1000.0 * finite("ttft", value) for value in result["ttfts"]]
    tpots_ms_all = per_request_tpot_ms(result)
    ttfts_ms = [value for value, ok in zip(ttfts_ms_all, success, strict=True) if ok]
    tpots_ms = [value for value, ok in zip(tpots_ms_all, success, strict=True) if ok]
    start_times = [finite("start_time", value) for value in result["start_times"]]
    arrival_span = max(start_times) - min(start_times)
    window = float(protocol["measurement_window_s"])
    arrival_ratio = arrival_span / window
    arrival_tolerance = float(protocol["arrival_window_tolerance_fraction"])
    require(1.0 - arrival_tolerance <= arrival_ratio <= 1.0 + arrival_tolerance, "arrival window drift")

    duration = finite("duration", result["duration"])
    offered_rate = float(sample["request_rate"])
    tpot_threshold = float(protocol["tpot_threshold_ms"])
    sustainable_ratio = float(protocol["sustainable_goodput_ratio"])
    sweep: dict[str, Any] = {}
    for raw_threshold in protocol["ttft_thresholds_ms"]:
        threshold = float(raw_threshold)
        good = sum(
            ok and ttft <= threshold and tpot <= tpot_threshold
            for ok, ttft, tpot in zip(success, ttfts_ms_all, tpots_ms_all, strict=True)
        )
        goodput = good / duration
        ratio = goodput / offered_rate
        sweep[f"{threshold:g}"] = {
            "good_requests": good,
            "goodput_req_s": goodput,
            "goodput_over_offered": ratio,
            "sustainable": ratio >= sustainable_ratio,
        }
    max_threshold = f"{max(map(float, protocol['ttft_thresholds_ms'])):g}"
    require(
        abs(finite("request_goodput", result["request_goodput"]) - sweep[max_threshold]["goodput_req_s"])
        <= float(protocol["goodput_crosscheck_abs_tolerance"]),
        "vLLM goodput does not match the independent recomputation",
    )
    return {
        "accounting": accounting,
        "duration_s": duration,
        "arrival_span_s": arrival_span,
        "arrival_ratio": arrival_ratio,
        "request_throughput_req_s": finite("request_throughput", result["request_throughput"]),
        "request_throughput_over_offered": finite("request_throughput", result["request_throughput"]) / offered_rate,
        "ttft_p95_ms": percentile(ttfts_ms, 95),
        "ttft_p99_ms": percentile(ttfts_ms, 99),
        "tpot_p95_ms": percentile(tpots_ms, 95),
        "tpot_p99_ms": percentile(tpots_ms, 99),
        "slo_sweep": sweep,
    }


def audit_sample(
    attempt_dir: Path,
    plan: Mapping[str, Any],
    *,
    root_commit: str,
    vllm_commit: str,
    require_zero_failures: bool = True,
) -> dict[str, Any]:
    sample_id = str(plan["sample_id"])
    sample_dir = attempt_dir / "samples" / sample_id
    require(sample_dir.is_dir(), f"missing sample directory: {sample_id}")
    hashes = {
        name: verify_sidecar(sample_dir / f"{name}.sha256")
        for name in ("contract.json", "result.json", "analysis.json")
    }
    contract = load_json(sample_dir / "contract.json")
    result = load_json(sample_dir / "result.json")
    analysis = load_json(sample_dir / "analysis.json")
    status = load_json(sample_dir / "status.json")

    require(status["status"] == "completed_validated", f"{sample_id}: invalid status")
    require(status["returncode"] == 0, f"{sample_id}: nonzero benchmark return code")
    require(status["result_sha256"] == hashes["result.json"], f"{sample_id}: status result hash")
    require(status["analysis_sha256"] == hashes["analysis.json"], f"{sample_id}: status analysis hash")
    require(contract["contract_sha256"] == canonical_sample_contract_sha(contract), f"{sample_id}: contract hash")
    for key in ("sample_id", "allocation", "workload", "num_prompts", "request_rate", "seed"):
        require(contract[key] == plan[key], f"{sample_id}: frozen {key} drift")
    require(contract["git_commit"] == root_commit, f"{sample_id}: root commit drift")
    require(contract["vllm_source_commit"] == vllm_commit, f"{sample_id}: vLLM commit drift")
    require(result["sample_id"] == sample_id, f"{sample_id}: result ID drift")
    require(result["allocation"] == plan["allocation"], f"{sample_id}: result allocation drift")
    require(result["attempt_id"] == contract["attempt_id"], f"{sample_id}: result attempt drift")
    require(str(result["seed"]) == str(plan["seed"]), f"{sample_id}: result seed drift")
    require(int(result["num_prompts"]) == int(plan["num_prompts"]), f"{sample_id}: result denominator drift")
    require(result["contract_sha256"] == contract["contract_sha256"], f"{sample_id}: result contract drift")
    require(result["git_commit"] == root_commit, f"{sample_id}: result root commit drift")
    require(result["vllm_source_commit"] == vllm_commit, f"{sample_id}: result vLLM commit drift")

    independent = recompute_analysis(result, plan, contract["protocol"])
    if require_zero_failures:
        require(independent["accounting"]["failed"] == 0, f"{sample_id}: failed requests present")
    require(analysis["status"] == "completed_validated", f"{sample_id}: analysis status")
    require(analysis["sample_id"] == sample_id, f"{sample_id}: analysis ID")
    require(analysis["request_failure_policy"] == "count_as_slo_miss", f"{sample_id}: failure policy")
    for observed_key, independent_key in (
        ("observed_arrival_span_s", "arrival_span_s"),
        ("arrival_span_over_target", "arrival_ratio"),
        ("benchmark_duration_s", "duration_s"),
        ("request_throughput_req_s", "request_throughput_req_s"),
        ("request_throughput_over_offered", "request_throughput_over_offered"),
        ("ttft_p95_ms_recomputed", "ttft_p95_ms"),
        ("ttft_p99_ms_recomputed", "ttft_p99_ms"),
        ("tpot_p95_ms_recomputed", "tpot_p95_ms"),
        ("tpot_p99_ms_recomputed", "tpot_p99_ms"),
    ):
        require_close(f"{sample_id}.{observed_key}", analysis[observed_key], independent[independent_key])
    for threshold, expected in independent["slo_sweep"].items():
        observed = analysis["slo_sweep"][threshold]
        require(observed["good_requests"] == expected["good_requests"], f"{sample_id}: good count {threshold}")
        require(observed["sustainable"] == expected["sustainable"], f"{sample_id}: sustainable {threshold}")
        require_close(
            f"{sample_id}.goodput.{threshold}",
            observed["goodput_req_s"],
            expected["goodput_req_s"],
        )

    command = [str(value) for value in contract["command"]]
    require(command_value(command, "--num-prompts") == str(plan["num_prompts"]), f"{sample_id}: command denominator")
    require(command_value(command, "--num-warmups") == "120", f"{sample_id}: command warmup")
    require(
        float(command_value(command, "--request-rate")) == float(plan["request_rate"]), f"{sample_id}: command rate"
    )
    require(command_value(command, "--seed") == str(plan["seed"]), f"{sample_id}: command seed")
    require(not (sample_dir / "bench.log.partial").exists(), f"{sample_id}: partial log remains")
    require(not (sample_dir / "work").exists(), f"{sample_id}: work directory remains")
    return {
        "sample_id": sample_id,
        "allocation": plan["allocation"],
        "workload": plan["workload"],
        "offered_rate_req_s": float(plan["request_rate"]),
        "seed": int(plan["seed"]),
        "status": status["status"],
        "accounting": independent["accounting"],
        "runner_duration_s": status["runner_duration_s"],
        "benchmark_duration_s": independent["duration_s"],
        "arrival_span_over_target": independent["arrival_ratio"],
        "request_throughput_req_s": independent["request_throughput_req_s"],
        "ttft_p95_ms": independent["ttft_p95_ms"],
        "tpot_p95_ms": independent["tpot_p95_ms"],
        "goodput_req_s_ttft_250": independent["slo_sweep"]["250"]["goodput_req_s"],
        "goodput_req_s_ttft_3000": independent["slo_sweep"]["3000"]["goodput_req_s"],
        "slo_sweep": independent["slo_sweep"],
        "all_thresholds_sustainable": all(item["sustainable"] for item in independent["slo_sweep"].values()),
        "hashes": hashes,
    }


def audit_server(
    attempt_dir: Path,
    allocation: str,
    allocation_config: Mapping[str, Any],
) -> dict[str, Any]:
    sessions = [path for path in (attempt_dir / "servers" / allocation).iterdir() if path.is_dir()]
    require(len(sessions) == 1, f"{allocation}: expected exactly one cold-start server session")
    session = sessions[0]
    contract_sha = verify_sidecar(session / "contract.json.sha256")
    contract = load_json(session / "contract.json")
    status = load_json(session / "status.json")
    log_path = session / "server.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    expected = EXPECTED_ALLOCATIONS[allocation]

    require(contract["allocation"] == allocation, f"{allocation}: server allocation drift")
    command = [str(value) for value in contract["command"]]
    require(command_value(command, "--kv-cache-dtype") == expected["kv_cache_dtype"], f"{allocation}: KV dtype")
    require(
        command_value(command, "--mamba-ssm-cache-dtype") == expected["state_dtype"],
        f"{allocation}: state dtype",
    )
    require(
        command_value(command, "--compilation-config") == '{"cudagraph_mode":"PIECEWISE"}', f"{allocation}: graph mode"
    )
    require("--tensor-parallel-size" not in command, f"{allocation}: unexpected TP override")
    require(status["status"] == "stopped", f"{allocation}: final server status")
    require(status["returncode"] == 0, f"{allocation}: server return code")
    require(status["exception"] is None, f"{allocation}: server exception")
    require(finite(f"{allocation}.startup", status["startup_duration_s"]) < 600, f"{allocation}: startup timeout")
    for proof in allocation_config["required_log_substrings"]:
        require(str(proof) in log_text, f"{allocation}: missing log proof {proof!r}")
    require(log_text.count("GET /health HTTP/1.1") >= 2, f"{allocation}: missing post-benchmark health proof")
    fatal_signatures = {signature: signature in log_text for signature in FATAL_SERVER_SIGNATURES}
    require(not any(fatal_signatures.values()), f"{allocation}: fatal server signature")
    return {
        "allocation": allocation,
        "session_id": session.name,
        "status": status["status"],
        "returncode": status["returncode"],
        "startup_duration_s": status["startup_duration_s"],
        "contract_sha256": contract_sha,
        "server_log_sha256": sha256_file(log_path),
        "health_200_count": log_text.count("GET /health HTTP/1.1"),
        "fatal_signatures": fatal_signatures,
        "stale_cubin_reload_warning_count": log_text.count("Failed to reload cubin file"),
        "precision_proofs": list(allocation_config["required_log_substrings"]),
    }


def fallacy_scan() -> list[dict[str, str]]:
    return [
        {
            "fallacy": "Simpson's paradox",
            "severity": "NOTE",
            "detail": "One workload/rate/seed; no aggregation claim is made.",
        },
        {
            "fallacy": "Ecological fallacy",
            "severity": "NOTE",
            "detail": "The unit is an allocation-workload-rate-seed sample; no request-level inference is made.",
        },
        {
            "fallacy": "Berkson's paradox",
            "severity": "CAUTION",
            "detail": "A single GPU, model, synthetic workload, and feasible rate are intentionally selected for MVEx.",
        },
        {
            "fallacy": "Collider bias",
            "severity": "NOTE",
            "detail": "No adjusted regression or conditioned causal model is used.",
        },
        {
            "fallacy": "Base-rate neglect",
            "severity": "NOTE",
            "detail": "No diagnostic classification metric is interpreted.",
        },
        {
            "fallacy": "Regression to the mean",
            "severity": "NOTE",
            "detail": "There is no pre/post selection on an extreme outcome.",
        },
        {
            "fallacy": "Survivorship bias",
            "severity": "SOLID",
            "detail": "All 4 planned samples and all 7,200 measurement requests are retained.",
        },
        {
            "fallacy": "Look-elsewhere effect",
            "severity": "CAUTION",
            "detail": "Five TTFT thresholds are shown; none is promoted as a confirmatory effect.",
        },
        {
            "fallacy": "Garden of forking paths",
            "severity": "NOTE",
            "detail": "The matrix, seed, rate, denominator, and SLO sweep were frozen before launch.",
        },
        {
            "fallacy": "Correlation is not causation",
            "severity": "CAUTION",
            "detail": "Orthogonal treatments are executed, but n=1 MVEx is not used for a causal performance claim.",
        },
        {
            "fallacy": "Reverse causality",
            "severity": "NOTE",
            "detail": "No observational directional association is interpreted.",
        },
    ]


def build_validation_markdown(report: Mapping[str, Any]) -> str:
    rows = "\n".join(
        "| {allocation} | {completed}/{expected} | {throughput:.3f} | {ttft:.2f} | {tpot:.2f} | {goodput:.3f} |".format(
            allocation=sample["allocation"],
            completed=sample["accounting"]["completed"],
            expected=sample["accounting"]["expected"],
            throughput=sample["request_throughput_req_s"],
            ttft=sample["ttft_p95_ms"],
            tpot=sample["tpot_p95_ms"],
            goodput=sample["goodput_req_s_ttft_250"],
        )
        for sample in report["samples"]
    )
    warnings = report["anomalies"]
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-11
- Verification Status: ANALYZED
- Version Label: joint_precision_four_config_mvex_r2_validation_v1
- Integrity Pass Date: {report["generated_at_utc"]}

## Validation Report

- **Source**: `{report["attempt_id"]}`
- **Gate 1 Verdict**: `{report["gate_1_verdict"]}`
- **Evidence Status**: `UNVERIFIED` diagnostic evidence
- **Overall Confidence**: `CAUTION`
- **Reproducibility**: `CANNOT_VERIFY` at single-seed MVEx

### Integrity Findings

The frozen set is complete: 4/4 samples, 7,200/7,200 measurement requests,
480/480 declared warmup requests, zero failed requests, zero missing or extra
sample directories, and launcher exit code 0. All {report["raw_sidecars"]["count"]}
raw SHA-256 sidecars match. Every allocation used one distinct cold-start server
session, passed its exact state/KV/PIECEWISE log proofs, remained healthy after
benchmarking, and stopped with return code 0.

| Allocation | Requests | Throughput req/s | P95 TTFT ms | P95 TPOT ms | Goodput @ TTFT 250 ms |
|---|---:|---:|---:|---:|---:|
{rows}

### Warnings

- {warnings[0]}
- {warnings[1]}

### Fallacy Scan

- **Coverage**: 11/11

This MVEx is a pipeline and denominator gate, not an effect-estimation study.
With one seed, one synthetic workload, and one feasible rate, no confidence
interval, p value, multiple-comparison decision, controller advantage, or
paper-level quantitative claim is valid. Calibration must use the frozen
multi-seed/rate matrix before profile construction.

### Promotion Decision

Gate 1 passes and authorizes the predeclared calibration slice. The run artifact
remains `UNVERIFIED`; it is not upgraded to `VERIFIED` because no applicable
reproducibility re-run or multi-seed statistical comparison has been completed.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--frozen-contract", type=Path, required=True)
    parser.add_argument("--expected-root-commit", required=True)
    parser.add_argument("--expected-vllm-commit", required=True)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--remote-archive-path", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_dir = args.artifact_dir.resolve()
    attempt_dir = artifact_dir / "attempt"
    launch_dir = artifact_dir / "launch"
    frozen_path = args.frozen_contract.resolve()
    frozen_sha = verify_sidecar(frozen_path.with_suffix(frozen_path.suffix + ".sha256"))
    frozen = load_json(frozen_path)
    attempt_id = str(frozen["attempt_id"])
    parent_attempt = str(frozen["parent_attempt"])

    archive_path = args.archive_path.resolve()
    require(archive_path.is_file(), "downloaded archive is missing")
    require(sha256_file(archive_path) == args.archive_sha256, "downloaded archive SHA mismatch")
    require(attempt_dir.is_dir() and launch_dir.is_dir(), "attempt or launch directory is missing")
    sidecars = sorted(attempt_dir.rglob("*.sha256"))
    sidecar_hashes = {str(path.relative_to(artifact_dir)).replace("\\", "/"): verify_sidecar(path) for path in sidecars}
    require(len(sidecars) == 19, f"expected 19 raw sidecars, found {len(sidecars)}")

    attempt = load_json(attempt_dir / "attempt_contract.json")
    environment = load_json(attempt_dir / "environment.json")
    summary = load_json(attempt_dir / "summary.json")
    require(attempt["attempt_id"] == attempt_id, "attempt ID drift")
    require(attempt["parent_attempt"] == parent_attempt, "parent attempt drift")
    require(attempt["git_commit"] == args.expected_root_commit, "attempt root commit drift")
    require(attempt["vllm_source_commit"] == args.expected_vllm_commit, "attempt vLLM commit drift")
    require(attempt["config_sha256"] == frozen["code"]["config_sha256"], "config hash drift")
    require(attempt["phase"]["name"] == "mvex", "phase drift")
    require(attempt["phase"]["allocations"] == list(EXPECTED_ALLOCATIONS), "allocation order drift")
    require(attempt["phase"]["seeds"] == [7], "seed drift")
    require(attempt["phase"]["workload_rates"] == {"random": [30.0]}, "workload/rate drift")
    require(
        environment["root_git"] == {"clean": True, "commit": args.expected_root_commit, "status": ""},
        "environment root Git drift",
    )
    require(environment["vllm_source_commit"] == args.expected_vllm_commit, "environment vLLM drift")
    require(environment["model_config_sha256"] == frozen["model"]["config_sha256"], "model hash drift")

    plan = list(attempt["plan"])
    plan_by_id = {str(item["sample_id"]): item for item in plan}
    expected_ids = list(frozen["matrix"]["sample_ids"])
    require(list(plan_by_id) == expected_ids, "sample plan order or IDs drift")
    observed_ids = sorted(path.name for path in (attempt_dir / "samples").iterdir() if path.is_dir())
    require(observed_ids == sorted(expected_ids), "missing or extra sample directories")
    require(summary["counts"] == {"completed_validated": 4}, "summary denominator drift")
    require([item["sample_id"] for item in summary["samples"]] == expected_ids, "summary sample IDs drift")
    require(all(item["status"] == "completed_validated" for item in summary["samples"]), "summary status drift")

    invocations = list((attempt_dir / "invocations").glob("*.json"))
    require(len(invocations) == 1, "expected one invocation record")
    invocation = load_json(invocations[0])
    require(invocation["resume"] is False and invocation["max_samples"] is None, "attempt was resumed or sliced")
    argv = [str(value) for value in invocation["argv"]]
    require(command_value(argv, "--attempt-id") == attempt_id, "invocation attempt ID")
    require(command_value(argv, "--parent-attempt") == parent_attempt, "invocation parent")
    require(command_value(argv, "--phase") == "mvex", "invocation phase")

    require((launch_dir / "exit_code").read_text(encoding="ascii").strip() == "0", "launcher exit code")
    run_log = (launch_dir / "run.log").read_text(encoding="utf-8")
    require('{"completed_validated": 4}' in run_log, "launcher log summary missing")
    started = datetime.fromisoformat((launch_dir / "started_at").read_text(encoding="ascii").strip())
    finished = datetime.fromisoformat((launch_dir / "finished_at").read_text(encoding="ascii").strip())
    require(finished > started, "launcher timestamps are not monotonic")

    samples = [
        audit_sample(
            attempt_dir,
            plan_by_id[sample_id],
            root_commit=args.expected_root_commit,
            vllm_commit=args.expected_vllm_commit,
        )
        for sample_id in expected_ids
    ]
    sample_contracts = {
        sample_id: load_json(attempt_dir / "samples" / sample_id / "contract.json") for sample_id in expected_ids
    }
    servers = [
        audit_server(attempt_dir, allocation, sample_contracts[f"{allocation}__random__r30__s7"]["allocation_config"])
        for allocation in EXPECTED_ALLOCATIONS
    ]
    total_expected = sum(item["accounting"]["expected"] for item in samples)
    total_completed = sum(item["accounting"]["completed"] for item in samples)
    total_failed = sum(item["accounting"]["failed"] for item in samples)
    require(total_expected == total_completed == 7200 and total_failed == 0, "aggregate request denominator drift")
    require(not list(attempt_dir.rglob("*.partial")), "partial artifact remains")

    scans = fallacy_scan()
    require(len(scans) == 11, "fallacy scan must cover 11 categories")
    report = {
        "schema_version": 1,
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "origin_date": "2026-08-11",
            "verification_status": "ANALYZED",
            "version_label": "joint_precision_four_config_mvex_r2_validation_v1",
        },
        "generated_at_utc": utc_timestamp(),
        "attempt_id": attempt_id,
        "parent_attempt": parent_attempt,
        "classification": "diagnostic_minimum_viable_execution",
        "gate_1_verdict": "PASS",
        "evidence_status": "UNVERIFIED",
        "overall_confidence": "CAUTION",
        "root_commit": args.expected_root_commit,
        "vllm_commit": args.expected_vllm_commit,
        "verification_tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "frozen_contract": {"path": str(frozen_path), "sha256": frozen_sha},
        "archive_receipt": {
            "remote_path": args.remote_archive_path,
            "local_path": str(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "sha256": args.archive_sha256,
        },
        "launch": {
            "exit_code": 0,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_s": (finished - started).total_seconds(),
        },
        "denominator": {
            "planned_samples": 4,
            "completed_validated_samples": 4,
            "planned_measurement_requests": total_expected,
            "completed_measurement_requests": total_completed,
            "failed_measurement_requests": total_failed,
            "declared_warmup_requests": 480,
            "missing_samples": 0,
            "extra_samples": 0,
            "silent_exclusions": 0,
        },
        "raw_sidecars": {"count": len(sidecars), "verified": len(sidecars), "hashes": sidecar_hashes},
        "samples": samples,
        "servers": servers,
        "anomalies": [
            "The first full-precision cold start logged stale Triton cubin reload warnings after cache cleanup; vLLM rebuilt/loaded artifacts, reached health, completed the sample, and passed the post-benchmark health check.",
            "The vLLM controlled shutdown path logs EngineDeadError/force-kill cleanup noise, but each server status records exception=null and returncode=0 after a healthy sample.",
        ],
        "statistical_scope": {
            "unit_of_analysis": "one allocation-workload-rate-seed sample",
            "independent_repeats_per_allocation": 1,
            "effect_size": "not estimated",
            "confidence_interval": "not estimable",
            "p_values": "not applicable",
            "multiple_comparisons": "five SLO thresholds are diagnostic only",
        },
        "fallacy_scan_coverage": "11/11",
        "fallacy_scan": scans,
        "reproducibility": {
            "determinism_class": "environment_sensitive_seeded_serving_benchmark",
            "method": "not run at MVEx",
            "verdict": "CANNOT_VERIFY",
            "reason": "Gate 1 validates the execution path; the predeclared multi-seed calibration is the next reproducibility/statistical gate.",
        },
        "promotion": {
            "calibration_authorized": True,
            "paper_quantitative_use_authorized": False,
            "reason": "Gate 1 passed, but single-seed diagnostic evidence remains UNVERIFIED.",
        },
    }

    report_sha = write_json_with_hash(artifact_dir / "mvex_audit_report.json", report)
    validation_sha = write_text_with_hash(
        artifact_dir / "validation_report.md",
        build_validation_markdown(report),
    )
    manifest_paths = sorted(
        path
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name not in {"artifact_sha256_manifest.json", "artifact_sha256_manifest.json.sha256"}
    )
    manifest = {
        "schema_version": 1,
        "material_passport": report["material_passport"],
        "generated_at_utc": utc_timestamp(),
        "attempt_id": attempt_id,
        "scope": "Complete downloaded MVEx evidence plus independent Gate 1 audit",
        "files": [
            {
                "relative_path": str(path.relative_to(artifact_dir)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in manifest_paths
        ],
        "generated": {
            "mvex_audit_report.json": report_sha,
            "validation_report.md": validation_sha,
        },
    }
    manifest_sha = write_json_with_hash(artifact_dir / "artifact_sha256_manifest.json", manifest)
    print(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "gate_1_verdict": report["gate_1_verdict"],
                "samples": f"{len(samples)}/4",
                "measurement_requests": f"{total_completed}/{total_expected}",
                "failed_requests": total_failed,
                "raw_sidecars": f"{len(sidecars)}/{len(sidecars)}",
                "manifest_files": len(manifest_paths),
                "manifest_sha256": manifest_sha,
                "calibration_authorized": True,
                "paper_quantitative_use_authorized": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
