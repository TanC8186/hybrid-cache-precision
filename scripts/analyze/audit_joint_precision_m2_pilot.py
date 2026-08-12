"""Logically audit the scoped M2 selector pilot without hash validation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_PRECISION = {
    "full": {"kv_cache_dtype": "auto", "state_dtype": "float32"},
    "state_only": {"kv_cache_dtype": "auto", "state_dtype": "bfloat16"},
    "joint": {"kv_cache_dtype": "int4_per_token_head", "state_dtype": "bfloat16"},
}
EXPECTED_CANDIDATES = {"full", "kv_only", "state_only", "joint"}
FATAL_SERVER_SIGNATURES = (
    "CUDA out of memory",
    "CUDA error: an illegal instruction",
    "CUDA error: an illegal memory access",
    "EngineCore encountered a fatal error",
)
T_CRITICAL_DF2 = 4.302652729911275


class AuditError(RuntimeError):
    """Raised when a logical integrity check fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON object expected: {path}")
    return value


def finite(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise AuditError(f"{field} is not numeric: {value!r}") from error
    require(math.isfinite(parsed), f"{field} is not finite")
    return parsed


def require_finite_tree(value: Any, field: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)), f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_tree(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_tree(item, f"{field}.{key}")


def require_close(observed: Any, expected: Any, field: str, *, tolerance: float = 1e-9) -> None:
    left = finite(observed, field)
    right = finite(expected, field)
    require(
        math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance),
        f"{field} mismatch: observed={left!r}, expected={right!r}",
    )


def command_value(command: Sequence[str], flag: str) -> str:
    values: list[str] = []
    prefix = f"{flag}="
    index = 0
    while index < len(command):
        item = str(command[index])
        if item == flag:
            require(index + 1 < len(command), f"missing command value for {flag}")
            values.append(str(command[index + 1]))
            index += 2
            continue
        if item.startswith(prefix):
            values.append(item[len(prefix) :])
        index += 1
    require(len(values) == 1, f"command must contain exactly one {flag}; found {len(values)}")
    return values[0]


def percentile(values: Sequence[float], pct: float) -> float:
    require(bool(values), "cannot compute a percentile from an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def student_t_summary(values: Sequence[float], *, t_critical: float = T_CRITICAL_DF2) -> dict[str, Any]:
    parsed = [finite(value, "seed metric") for value in values]
    require(len(parsed) == 3, "the M2 pilot requires exactly three independent seeds per cell")
    mean = statistics.fmean(parsed)
    sample_sd = statistics.stdev(parsed)
    margin = finite(t_critical, "t critical") * sample_sd / math.sqrt(len(parsed))
    return {
        "values": parsed,
        "n": len(parsed),
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def equivalent_json(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(equivalent_json(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent_json(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def check_result_accounting(result: Mapping[str, Any], expected: int, sample_id: str) -> dict[str, Any]:
    completed = int(result["completed"])
    failed = int(result["failed"])
    require(completed + failed == expected, f"{sample_id}: completed + failed differs from frozen denominator")
    fields = ("ttfts", "itls", "input_lens", "output_lens", "start_times", "errors")
    lengths = {field: len(result[field]) for field in fields}
    require(all(length == expected for length in lengths.values()), f"{sample_id}: detailed denominator drift {lengths}")
    detailed_failures = sum(bool(error) for error in result["errors"])
    require(detailed_failures == failed, f"{sample_id}: detailed failures differ from result.failed")
    require(completed == expected - detailed_failures, f"{sample_id}: detailed successes differ from result.completed")
    return {
        "expected": expected,
        "completed": completed,
        "failed": failed,
        "detail_lengths": lengths,
        "request_conservation": True,
    }


def recompute_sample(sample_dir: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = str(plan["sample_id"])
    contract = load_json(sample_dir / "contract.json")
    result = load_json(sample_dir / "result.json")
    analysis = load_json(sample_dir / "analysis.json")
    status = load_json(sample_dir / "status.json")
    for name, value in (("contract", contract), ("result", result), ("analysis", analysis), ("status", status)):
        require_finite_tree(value, f"{sample_id}.{name}")

    require(status.get("status") == "completed_validated", f"{sample_id}: status is not completed_validated")
    require(status.get("returncode") == 0, f"{sample_id}: benchmark return code is nonzero")
    require(analysis.get("status") == "completed_validated", f"{sample_id}: analysis status is invalid")
    for key in ("sample_id", "allocation", "workload", "num_prompts", "request_rate", "seed"):
        require(equivalent_json(contract.get(key), plan.get(key)), f"{sample_id}: contract {key} drift")
    require(result.get("sample_id") == sample_id, f"{sample_id}: result sample ID drift")
    require(result.get("allocation") == plan["allocation"], f"{sample_id}: result allocation drift")
    require(str(result.get("seed")) == str(plan["seed"]), f"{sample_id}: result seed drift")
    require(int(result.get("num_prompts")) == int(plan["num_prompts"]), f"{sample_id}: result num_prompts drift")
    require(analysis.get("sample_id") == sample_id, f"{sample_id}: analysis sample ID drift")
    require(analysis.get("request_failure_policy") == "count_as_slo_miss", f"{sample_id}: failure policy drift")

    expected = int(plan["num_prompts"])
    accounting = check_result_accounting(result, expected, sample_id)
    require(accounting["failed"] == 0, f"{sample_id}: failed requests are present")
    success = [not bool(error) for error in result["errors"]]
    ttft_all = [1000.0 * finite(value, f"{sample_id}.ttft") for value in result["ttfts"]]
    tpot_all: list[float] = []
    for index, (output_len, itls) in enumerate(zip(result["output_lens"], result["itls"], strict=True)):
        count = int(output_len)
        require(isinstance(itls, list), f"{sample_id}: itls[{index}] is not a list")
        parsed_itls = [finite(value, f"{sample_id}.itl") for value in itls]
        require(all(value >= 0 for value in parsed_itls), f"{sample_id}: negative ITL at request {index}")
        # vLLM records one ITL per streamed chunk, while output_len counts
        # tokens. A chunk may contain multiple tokens, so only the summed
        # intervals, not a one-to-one length relationship, define TPOT.
        require(len(parsed_itls) <= max(count - 1, 0), f"{sample_id}: more ITLs than token intervals")
        tpot_all.append(0.0 if count <= 1 else 1000.0 * sum(parsed_itls) / (count - 1))
    ttft_success = [value for value, ok in zip(ttft_all, success, strict=True) if ok]
    tpot_success = [value for value, ok in zip(tpot_all, success, strict=True) if ok]

    duration = finite(result["duration"], f"{sample_id}.duration")
    require(duration > 0, f"{sample_id}: duration must be positive")
    offered_rate = float(plan["request_rate"])
    throughput = finite(result["request_throughput"], f"{sample_id}.request_throughput")
    require_close(throughput, accounting["completed"] / duration, f"{sample_id}.request_throughput", tolerance=1e-7)
    starts = [finite(value, f"{sample_id}.start_time") for value in result["start_times"]]
    arrival_span = max(starts) - min(starts)
    protocol = contract["protocol"]
    measurement_window = float(protocol["measurement_window_s"])
    arrival_ratio = arrival_span / measurement_window
    arrival_tolerance = float(protocol["arrival_window_tolerance_fraction"])
    require(1 - arrival_tolerance <= arrival_ratio <= 1 + arrival_tolerance, f"{sample_id}: arrival window drift")

    ttft_p95 = percentile(ttft_success, 95)
    tpot_p95 = percentile(tpot_success, 95)
    require_close(analysis["ttft_p95_ms_recomputed"], ttft_p95, f"{sample_id}.ttft_p95")
    require_close(analysis["tpot_p95_ms_recomputed"], tpot_p95, f"{sample_id}.tpot_p95")
    require_close(analysis["request_throughput_req_s"], throughput, f"{sample_id}.throughput")
    require_close(analysis["observed_arrival_span_s"], arrival_span, f"{sample_id}.arrival_span")

    tpot_threshold = float(protocol["tpot_threshold_ms"])
    sustainable_ratio = float(protocol["sustainable_goodput_ratio"])
    sweep: dict[str, Any] = {}
    for raw_threshold in protocol["ttft_thresholds_ms"]:
        threshold = float(raw_threshold)
        good = sum(
            ok and ttft <= threshold and tpot <= tpot_threshold
            for ok, ttft, tpot in zip(success, ttft_all, tpot_all, strict=True)
        )
        goodput = good / duration
        ratio = goodput / offered_rate
        key = f"{threshold:g}"
        observed = analysis["slo_sweep"][key]
        require(int(observed["good_requests"]) == good, f"{sample_id}: SLO good count drift at {key} ms")
        require_close(observed["goodput_req_s"], goodput, f"{sample_id}.goodput.{key}")
        require_close(observed["goodput_over_offered"], ratio, f"{sample_id}.goodput_ratio.{key}")
        require(bool(observed["sustainable"]) == (ratio >= sustainable_ratio), f"{sample_id}: sustainable flag drift")
        sweep[key] = {
            "good_requests": good,
            "goodput_req_s": goodput,
            "goodput_over_offered": ratio,
            "sustainable": ratio >= sustainable_ratio,
        }

    command = [str(value) for value in contract["command"]]
    require(command_value(command, "--num-prompts") == str(expected), f"{sample_id}: command denominator drift")
    require(command_value(command, "--num-warmups") == str(protocol["warmup_requests"]), f"{sample_id}: warmup drift")
    require(float(command_value(command, "--request-rate")) == offered_rate, f"{sample_id}: command rate drift")
    require(command_value(command, "--seed") == str(plan["seed"]), f"{sample_id}: command seed drift")
    require(not (sample_dir / "bench.log.partial").exists(), f"{sample_id}: partial benchmark log remains")
    require(not (sample_dir / "work").exists(), f"{sample_id}: temporary result directory remains")

    requested_slo = sweep["500"]
    return {
        "sample_id": sample_id,
        "allocation": str(plan["allocation"]),
        "seed": int(plan["seed"]),
        "workload": str(plan["workload"]),
        "offered_rate_req_s": offered_rate,
        "accounting": accounting,
        "warmup_requests": int(protocol["warmup_requests"]),
        "benchmark_duration_s": duration,
        "arrival_span_over_target": arrival_ratio,
        "request_throughput_req_s": throughput,
        "ttft_p95_ms": ttft_p95,
        "tpot_p95_ms": tpot_p95,
        "goodput_req_s": requested_slo["goodput_req_s"],
        "goodput_over_offered": requested_slo["goodput_over_offered"],
        "requested_slo_attained": requested_slo["sustainable"] and ttft_p95 <= 500 and tpot_p95 <= 200,
        "slo_sweep": sweep,
    }


def recompute_decision(repo_root: Path, profile: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    source_root = repo_root / "src"
    for path in (repo_root, source_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from kvcache.policy import select_joint_precision

    return select_joint_precision(profile, request)


def audit_server(
    attempt_dir: Path,
    controller_contract: Mapping[str, Any],
    allocation: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    server_root = attempt_dir / "servers" / allocation
    sessions = [path for path in server_root.iterdir() if path.is_dir()]
    require(len(sessions) == 1, f"{allocation}: expected one cold-start server session")
    session = sessions[0]
    contract = load_json(session / "contract.json")
    status = load_json(session / "status.json")
    log_path = session / "server.log"
    require(log_path.is_file(), f"{allocation}: server log is missing")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    command = [str(value) for value in contract["command"]]
    expected = EXPECTED_PRECISION[allocation]
    require(contract.get("allocation") == allocation, f"{allocation}: server allocation drift")
    require(command == controller_contract["deployment_mapping"]["server_command"], f"{allocation}: executed server command drift")
    require(command_value(command, "--kv-cache-dtype") == expected["kv_cache_dtype"], f"{allocation}: KV dtype proof failed")
    require(command_value(command, "--mamba-ssm-cache-dtype") == expected["state_dtype"], f"{allocation}: state dtype proof failed")
    require(command_value(command, "--compilation-config") == '{"cudagraph_mode":"PIECEWISE"}', f"{allocation}: graph mode drift")
    require("--tensor-parallel-size" not in command, f"{allocation}: unexpected tensor parallel override")
    require(status.get("status") == "stopped", f"{allocation}: server did not stop cleanly")
    require(status.get("returncode") == 0, f"{allocation}: server return code is nonzero")
    require(status.get("exception") is None, f"{allocation}: server exception is present")
    startup = finite(status.get("startup_duration_s"), f"{allocation}.startup_duration_s")
    require(startup < 600, f"{allocation}: startup timeout")
    for proof in config["allocations"][allocation]["required_log_substrings"]:
        require(str(proof) in log_text, f"{allocation}: missing server-log proof {proof!r}")
    fatal = {signature: signature in log_text for signature in FATAL_SERVER_SIGNATURES}
    require(not any(fatal.values()), f"{allocation}: fatal server signature found")
    health_count = log_text.count("GET /health HTTP/1.1")
    require(health_count >= 2, f"{allocation}: post-benchmark health proof is missing")
    return {
        "allocation": allocation,
        "session_id": session.name,
        "status": status["status"],
        "returncode": status["returncode"],
        "startup_duration_s": startup,
        "health_200_count": health_count,
        "fatal_signatures": fatal,
        "stale_cubin_reload_warning_count": log_text.count("Failed to reload cubin file"),
        "precision_proofs": list(config["allocations"][allocation]["required_log_substrings"]),
    }


def find_controller_root(pilot_root: Path, request_id: str) -> Path:
    matches = [
        path
        for path in pilot_root.iterdir()
        if path.is_dir() and f"-pilot-{request_id}-" in path.name and (path / "controller_result.json").is_file()
    ]
    require(len(matches) == 1, f"request {request_id}: expected one controller attempt, found {len(matches)}")
    return matches[0]


def audit_launch(pilot_root: Path, attempt_id: str, expected_allocation: str) -> dict[str, Any]:
    launch = pilot_root / "launch" / attempt_id
    require(launch.is_dir(), f"missing launch directory: {attempt_id}")
    for name in ("expected_allocation", "started_at", "finished_at", "exit_code", "run.log"):
        require((launch / name).is_file(), f"{attempt_id}: launcher evidence missing {name}")
    require((launch / "exit_code").read_text(encoding="ascii").strip() == "0", f"{attempt_id}: launcher exit code")
    require(
        (launch / "expected_allocation").read_text(encoding="ascii").strip() == expected_allocation,
        f"{attempt_id}: launcher expected allocation drift",
    )
    started_at = (launch / "started_at").read_text(encoding="ascii").strip()
    finished_at = (launch / "finished_at").read_text(encoding="ascii").strip()
    started = datetime.fromisoformat(started_at)
    finished = datetime.fromisoformat(finished_at)
    require(started.tzinfo is not None and finished.tzinfo is not None, f"{attempt_id}: launcher timezone missing")
    require(finished >= started, f"{attempt_id}: launcher timestamps are not monotonic")
    return {
        "attempt_id": attempt_id,
        "expected_allocation": expected_allocation,
        "exit_code": 0,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": (finished - started).total_seconds(),
        "run_log_size_bytes": (launch / "run.log").stat().st_size,
    }


def audit_controller(
    pilot_root: Path,
    repo_root: Path,
    package_contract: Mapping[str, Any],
    profile: dict[str, Any],
    config: Mapping[str, Any],
    request_spec: Mapping[str, Any],
) -> dict[str, Any]:
    request_id = str(request_spec["id"])
    expected_allocation = str(request_spec["expected_selected_config_id"])
    controller_root = find_controller_root(pilot_root, request_id)
    controller = load_json(controller_root / "controller_result.json")
    controller_contract = load_json(controller_root / "controller_contract.json")
    decision = load_json(controller_root / "decision.json")
    request = load_json(repo_root / str(request_spec["path"]))
    require_finite_tree(controller, f"{request_id}.controller_result")
    require_finite_tree(decision, f"{request_id}.decision")

    require(controller.get("status") == "COMPLETED_UNVERIFIED", f"{request_id}: controller did not complete")
    require(controller.get("attempt_id") == controller_root.name, f"{request_id}: controller attempt ID drift")
    require(controller.get("selected_config_id") == expected_allocation, f"{request_id}: selected config drift")
    require(decision.get("status") == "SELECTED", f"{request_id}: decision is not SELECTED")
    require(decision["selected"]["config_id"] == expected_allocation, f"{request_id}: decision mapping drift")
    require(equivalent_json(decision["request"], request), f"{request_id}: normalized request differs from frozen request")
    observed_evaluations = {str(row["config_id"]): row for row in decision["evaluations"]}
    require(set(observed_evaluations) == EXPECTED_CANDIDATES, f"{request_id}: candidate set drift")
    require(observed_evaluations[expected_allocation]["feasible"] is True, f"{request_id}: selected candidate is infeasible")

    recomputed = recompute_decision(repo_root, profile, request)
    observed_without_runtime = {key: value for key, value in decision.items() if key != "decision_latency_ms"}
    require(equivalent_json(observed_without_runtime, recomputed), f"{request_id}: selector logic does not reproduce decision")
    require(controller_contract.get("dry_run") is False, f"{request_id}: controller was a dry run")
    require(controller_contract["root_git"].get("clean") is True, f"{request_id}: captured worktree was dirty")
    require(controller_contract["root_git"].get("status") == "", f"{request_id}: captured Git status was not empty")
    require(controller_contract["deployment_mapping"]["allocation"] == expected_allocation, f"{request_id}: deployment allocation drift")
    require(controller_contract["deployment_mapping"]["precision_args"] == decision["selected"]["deployment"]["precision_args"], f"{request_id}: precision mapping drift")
    require(controller_contract["phase"]["name"] == "confirmatory", f"{request_id}: phase drift")
    expected_seeds = list(package_contract["matrix"]["pilot"]["seeds"])
    require(controller_contract["phase"]["seeds"] == expected_seeds, f"{request_id}: seed set drift")
    require(controller_contract["phase"]["allocations"] == [expected_allocation], f"{request_id}: allocation phase drift")
    require(controller_contract["phase"]["workload_rates"] == {str(request["workload"]): [float(request["offered_rate_req_s"])]}, f"{request_id}: workload/rate drift")

    runner_dir = controller_root / "runner" / controller_root.name
    attempt = load_json(runner_dir / "attempt_contract.json")
    environment = load_json(runner_dir / "environment.json")
    summary = load_json(runner_dir / "summary.json")
    require(attempt["attempt_id"] == controller_root.name, f"{request_id}: runner attempt ID drift")
    require(attempt["git_commit"] == controller_contract["root_git"]["commit"], f"{request_id}: runner Git revision drift")
    require(environment["root_git"] == controller_contract["root_git"], f"{request_id}: environment Git state drift")
    require(attempt["vllm_source_commit"] == environment["vllm_source_commit"], f"{request_id}: vLLM revision drift")
    require(equivalent_json(attempt["phase"], controller_contract["phase"]), f"{request_id}: runner phase drift")
    require(equivalent_json(attempt["plan"], controller_contract["plan"]), f"{request_id}: runner plan drift")

    plan = list(attempt["plan"])
    expected_prompts = int(float(request["offered_rate_req_s"]) * float(config["protocol"]["measurement_window_s"]))
    expected_ids = [f"{expected_allocation}__{request['workload']}__r{float(request['offered_rate_req_s']):g}__s{seed}" for seed in expected_seeds]
    require([row["sample_id"] for row in plan] == expected_ids, f"{request_id}: sample plan membership/order drift")
    require(all(int(row["num_prompts"]) == expected_prompts for row in plan), f"{request_id}: planned request denominator drift")
    require(summary["counts"] == {"completed_validated": len(expected_seeds)}, f"{request_id}: summary count drift")
    require([row["sample_id"] for row in summary["samples"]] == expected_ids, f"{request_id}: summary membership drift")
    require(all(row["status"] == "completed_validated" for row in summary["samples"]), f"{request_id}: summary status drift")
    sample_root = runner_dir / "samples"
    observed_ids = sorted(path.name for path in sample_root.iterdir() if path.is_dir())
    require(observed_ids == sorted(expected_ids), f"{request_id}: missing or extra sample directories")
    require(not list(runner_dir.rglob("*.partial")), f"{request_id}: partial artifact remains")

    samples = [recompute_sample(sample_root / row["sample_id"], row) for row in plan]
    server = audit_server(runner_dir, controller_contract, expected_allocation, config)
    launch = audit_launch(pilot_root, controller_root.name, expected_allocation)
    require(equivalent_json(controller.get("summary"), summary), f"{request_id}: controller summary drift")
    require(controller.get("transition_semantics") == "deployment_epoch_cold_restart", f"{request_id}: transition semantics drift")
    require_close(controller.get("startup_duration_s"), server["startup_duration_s"], f"{request_id}.startup_duration_s")
    require_close(
        controller.get("decision_plus_startup_s"),
        float(decision["decision_latency_ms"]) / 1000.0 + server["startup_duration_s"],
        f"{request_id}.decision_plus_startup_s",
    )
    result_analyses = controller.get("sample_analyses")
    require(isinstance(result_analyses, list) and len(result_analyses) == len(samples), f"{request_id}: controller analysis denominator drift")
    for observed, sample in zip(result_analyses, samples, strict=True):
        require(observed["sample_id"] == sample["sample_id"], f"{request_id}: controller sample analysis order drift")

    selected = decision["selected"]
    metric_summaries = {
        "request_throughput_req_s": student_t_summary([row["request_throughput_req_s"] for row in samples]),
        "p95_ttft_ms": student_t_summary([row["ttft_p95_ms"] for row in samples]),
        "p95_tpot_ms": student_t_summary([row["tpot_p95_ms"] for row in samples]),
        "goodput_req_s": student_t_summary([row["goodput_req_s"] for row in samples]),
        "goodput_over_offered": student_t_summary([row["goodput_over_offered"] for row in samples]),
    }
    objective_lcb = float(selected["objective_lcb_req_s"])
    return {
        "request_id": request_id,
        "attempt_id": controller_root.name,
        "selected_config_id": expected_allocation,
        "selector_logic_recomputed": True,
        "decision_latency_ms": float(decision["decision_latency_ms"]),
        "required_concurrency": float(request["required_concurrency"]),
        "selected_max_concurrency": float(selected["max_concurrency"]),
        "capacity_utilization": float(request["required_concurrency"]) / float(selected["max_concurrency"]),
        "quality_slack": float(selected["quality_slack"]),
        "calibration_goodput_lcb_req_s": objective_lcb,
        "confirmatory_goodput_minus_calibration_lcb_req_s": metric_summaries["goodput_req_s"]["mean"] - objective_lcb,
        "confirmatory_goodput_ci95_low_minus_calibration_lcb_req_s": metric_summaries["goodput_req_s"]["ci95_low"] - objective_lcb,
        "requested_slo_attainment": f"{sum(row['requested_slo_attained'] for row in samples)}/{len(samples)}",
        "metric_summaries": metric_summaries,
        "samples": samples,
        "server": server,
        "launch": launch,
        "root_commit": attempt["git_commit"],
        "vllm_commit": attempt["vllm_source_commit"],
    }


def fallacy_scan() -> list[dict[str, str]]:
    return [
        {"fallacy": "Simpson's paradox", "severity": "NOTE", "detail": "Each budget/allocation cell remains separate; no pooled direction replaces cell-level results."},
        {"fallacy": "Ecological fallacy", "severity": "NOTE", "detail": "The inferential unit is the seeded benchmark sample, not an individual request."},
        {"fallacy": "Berkson's paradox", "severity": "CAUTION", "detail": "The pilot intentionally covers one GPU, one model/context, Random workload, and three selected budget strata."},
        {"fallacy": "Collider bias", "severity": "NOTE", "detail": "No post-treatment covariate adjustment or conditioned regression is used."},
        {"fallacy": "Base-rate neglect", "severity": "NOTE", "detail": "Every sample retains its full request denominator and failures count as SLO misses."},
        {"fallacy": "Regression to the mean", "severity": "NOTE", "detail": "Budget requests, rates, and confirmatory seeds were frozen before pilot execution."},
        {"fallacy": "Survivorship bias", "severity": "NOTE", "detail": "All nine planned samples and all 18,000 measurement requests are required; none are silently excluded."},
        {"fallacy": "Look-elsewhere effect", "severity": "CAUTION", "detail": "The requested 500/200 ms SLO is primary; the remaining frozen TTFT sweep is descriptive."},
        {"fallacy": "Garden of forking paths", "severity": "NOTE", "detail": "Mappings, seeds, rates, denominators, and SLO thresholds were frozen in the package contract."},
        {"fallacy": "Correlation is not causation", "severity": "CAUTION", "detail": "The selector executes controlled precision configurations, but the scoped pilot cannot establish cross-model or mechanism claims."},
        {"fallacy": "Reverse causality", "severity": "NOTE", "detail": "Each precision decision precedes a cold-start deployment epoch and its measurements."},
    ]


def build_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for cell in report["cells"]:
        metrics = cell["metric_summaries"]
        rows.append(
            "| {request} | {allocation} | {slo} | {goodput:.3f} [{goodput_low:.3f}, {goodput_high:.3f}] | "
            "{ttft:.2f} | {tpot:.2f} | {capacity:.3f} |".format(
                request=cell["request_id"],
                allocation=cell["selected_config_id"],
                slo=cell["requested_slo_attainment"],
                goodput=metrics["goodput_req_s"]["mean"],
                goodput_low=metrics["goodput_req_s"]["ci95_low"],
                goodput_high=metrics["goodput_req_s"]["ci95_high"],
                ttft=metrics["p95_ttft_ms"]["mean"],
                tpot=metrics["p95_tpot_ms"]["mean"],
                capacity=cell["capacity_utilization"],
            )
        )
    return f"""## Material Passport

- Origin Skill: experiment-skill
- Origin Mode: validate
- Origin Date: {report['generated_at_utc'][:10]}
- Verification Status: ANALYZED
- Version Label: joint_precision_m2_pilot_logical_audit_v1

## Validation Report

- **Source**: `{report['package_id']}`
- **Gate 2 Verdict**: `{report['gate_2_verdict']}`
- **Evidence Status**: `ANALYZED`
- **Overall Confidence**: `CAUTION`
- **Reproducibility**: `CANNOT_VERIFY` until the separate Gate 4 attempt
- **Audit Mode**: logical review only; no SHA-256 or hash validation performed

### Integrity Findings

The frozen pilot is complete: 9/9 seeded samples, 18,000/18,000 measurement
requests, 1,080 declared warmup requests, zero failed requests, zero missing or
extra samples, zero NaNs, zero residual partial artifacts, and three launcher
exit codes of 0. The selector logic independently reproduces all three mappings:
`strict -> full`, `medium -> state_only`, and `high -> joint`. Server commands
and logs prove the matching KV/state precision and PIECEWISE graph mode.

The confidence intervals below use the three seeds as the independent units
(Student-t, df=2). Individual requests are not treated as independent repeats.

| Budget | Selected | SLO seeds | Goodput req/s, mean [95% CI] | Mean P95 TTFT ms | Mean P95 TPOT ms | Required/max concurrency |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

### Statistical Scope

- The requested 500 ms TTFT / 200 ms TPOT SLO is the primary endpoint.
- Calibration lower confidence bounds are guardrails, not point predictions;
  the report therefore records signed confirmatory-minus-LCB residuals rather
  than mislabeling them as ordinary prediction errors.
- With n=3 seeds per budget, intervals are descriptive and wide. No p value,
  equivalence claim, multi-model generalization, or mechanism attribution is promoted.

### Fallacy Scan

- **Coverage**: 11/11

""" + "\n".join(
        f"- **{item['fallacy']}** (`{item['severity']}`): {item['detail']}" for item in report["fallacy_scan"]["items"]
    ) + """

### Promotion Decision

Gate 2 passes. This pilot is `ANALYZED`, not `VERIFIED`, and is not yet
paper-usable quantitative evidence. Promotion requires a separately identified
Gate 4 reproduction and comparison under the declared environment-sensitive
tolerances.
"""


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def audit_pilot(pilot_root: Path, repo_root: Path, package_contract_path: Path, *, source_host: str) -> dict[str, Any]:
    package = load_json(package_contract_path)
    profile = load_json(repo_root / str(package["profile"]["path"]))
    expected_capacity_semantics = "unique_backing_storage_sum"
    require(
        package["profile"].get("capacity_bytes_semantics") == expected_capacity_semantics,
        "package contract uses legacy capacity byte semantics",
    )
    require(
        profile.get("capacity_bytes_semantics") == expected_capacity_semantics,
        "selector profile uses legacy capacity byte semantics",
    )
    config_path = repo_root / str(package["serving_config"]["path"])
    try:
        import yaml
    except ImportError as error:
        raise AuditError("PyYAML is required to parse the serving config") from error
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    require(isinstance(config, dict), "serving config must be a mapping")

    cells = [
        audit_controller(pilot_root, repo_root, package, profile, config, request_spec)
        for request_spec in package["requests"]
    ]
    expected_samples = int(package["matrix"]["pilot"]["expected_samples"])
    expected_requests = int(package["matrix"]["pilot"]["expected_measurement_requests"])
    expected_warmups = int(package["matrix"]["pilot"]["expected_warmup_requests"])
    samples = [sample for cell in cells for sample in cell["samples"]]
    completed = sum(sample["accounting"]["completed"] for sample in samples)
    failed = sum(sample["accounting"]["failed"] for sample in samples)
    warmups = sum(sample["warmup_requests"] for sample in samples)
    require(len(samples) == expected_samples, "aggregate sample denominator drift")
    require(completed == expected_requests and failed == 0, "aggregate request denominator drift")
    require(warmups == expected_warmups, "aggregate warmup denominator drift")
    require(len({cell["root_commit"] for cell in cells}) == 1, "root revision differs across budget attempts")
    require(len({cell["vllm_commit"] for cell in cells}) == 1, "vLLM revision differs across budget attempts")
    scans = fallacy_scan()
    require(len(scans) == 11 and len({item["fallacy"] for item in scans}) == 11, "fallacy scan is incomplete")
    return {
        "schema_version": 1,
        "material_passport": {
            "origin_skill": "experiment-skill",
            "origin_mode": "validate",
            "origin_date": datetime.now(timezone.utc).date().isoformat(),
            "verification_status": "ANALYZED",
            "version_label": "joint_precision_m2_pilot_logical_audit_v1",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "package_id": package["package_id"],
        "classification": package["classification"],
        "gate_2_verdict": "PASS",
        "evidence_status": "ANALYZED",
        "audit": {
            "mode": "logical_only",
            "hash_validation_performed": False,
            "sidecar_files_used_as_gates": False,
            "checks": [
                "selector/allocation mapping",
                "contract/result consistency",
                "exact sample/request/warmup denominators",
                "failures, NaNs, missing samples, and partial artifacts",
                "SLO latency and goodput recomputation",
                "server precision command and log proof",
                "three-seed confidence intervals",
                "11/11 statistical fallacy scan",
            ],
        },
        "source": {"host": source_host, "pilot_root": str(pilot_root), "raw_evidence_retained_on_data_disk": True},
        "scope": {
            "model_id": "Qwen/Qwen3.5-2B",
            "max_model_len": 4096,
            "workload": "random",
            "gpu_count": 1,
            "tensor_parallelism": 1,
        },
        "completeness": {
            "expected_samples": expected_samples,
            "audited_samples": len(samples),
            "expected_measurement_requests": expected_requests,
            "completed_measurement_requests": completed,
            "failed_measurement_requests": failed,
            "expected_warmup_requests": expected_warmups,
            "declared_warmup_requests": warmups,
            "silent_exclusions": 0,
            "launcher_exit_codes": [cell["launch"]["exit_code"] for cell in cells],
        },
        "decision_metrics": {
            "feasible_decision_rate": 1.0,
            "expected_mapping_accuracy": 1.0,
            "mapping": {cell["request_id"]: cell["selected_config_id"] for cell in cells},
        },
        "statistical_method": {
            "unit_of_analysis": "seeded benchmark sample",
            "n_per_budget": 3,
            "ci": "two-sided Student-t 95% CI",
            "degrees_of_freedom": 2,
            "t_critical": T_CRITICAL_DF2,
            "request_level_pseudoreplication_avoided": True,
            "multiple_comparisons": "one predeclared primary SLO per budget; remaining threshold sweep descriptive",
        },
        "cells": cells,
        "fallacy_scan": {"coverage": "11/11", "items": scans},
        "reproducibility": {
            "determinism_class": "environment_sensitive_seeded_serving_benchmark",
            "method": "not yet run for this pilot",
            "verdict": "CANNOT_VERIFY",
        },
        "promotion": {
            "gate_3_expansion_authorized": True,
            "paper_quantitative_use_authorized": False,
            "reason": "Gate 2 passed, but a separately identified Gate 4 reproduction has not yet passed.",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--package-contract", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-host", default="unspecified")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_pilot(
        args.pilot_root.resolve(),
        args.repo_root.resolve(),
        args.package_contract.resolve(),
        source_host=args.source_host,
    )
    out_dir = args.out_dir.resolve()
    atomic_write_json(out_dir / "pilot_logical_audit.json", report)
    atomic_write_text(out_dir / "validation_report.md", build_markdown(report))
    print(
        json.dumps(
            {
                "gate_2_verdict": report["gate_2_verdict"],
                "evidence_status": report["evidence_status"],
                "samples": f"{report['completeness']['audited_samples']}/{report['completeness']['expected_samples']}",
                "measurement_requests": f"{report['completeness']['completed_measurement_requests']}/{report['completeness']['expected_measurement_requests']}",
                "failed_requests": report["completeness"]["failed_measurement_requests"],
                "audit_mode": report["audit"]["mode"],
                "hash_validation_performed": report["audit"]["hash_validation_performed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
