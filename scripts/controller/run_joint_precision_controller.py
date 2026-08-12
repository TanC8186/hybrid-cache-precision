"""Select a joint precision configuration and execute its frozen serving slice."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from kvcache.policy import CAPACITY_BYTES_SEMANTICS, NoFeasibleCandidate, PolicyInputError, select_joint_precision
from scripts.bench.run_steady_state import (
    ExperimentError,
    build_sample_plan,
    build_server_command,
    get_git_state,
    load_config,
    resolve_phase,
    sha256_file,
    write_json_with_hash,
)
from scripts.bench.run_steady_state import (
    main as run_steady_state,
)
from scripts.controller.build_joint_precision_profile import (
    ProfileBuildError,
    verify_profile_evidence,
)


class ControllerError(RuntimeError):
    """Raised when a decision cannot be mapped to the frozen serving runner."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ControllerError(f"JSON root must be an object: {path}")
    return value


def review_profile_evidence_logically(profile: Mapping[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    """Require referenced evidence to exist and parse, without digest checks."""

    raw_records = profile.get("evidence")
    if not isinstance(raw_records, list) or not raw_records:
        raise ProfileBuildError("profile.evidence must be a non-empty array")
    resolved_root = repo_root.resolve()
    reviewed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ProfileBuildError(f"profile.evidence[{index}] must be an object")
        evidence_id = raw_record.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ProfileBuildError(f"profile.evidence[{index}].evidence_id must be a non-empty string")
        if evidence_id in seen:
            raise ProfileBuildError(f"duplicate evidence_id: {evidence_id}")
        seen.add(evidence_id)
        raw_path = raw_record.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ProfileBuildError(f"profile.evidence.{evidence_id}.path must be a non-empty string")
        declared = Path(raw_path)
        if declared.is_absolute():
            raise ProfileBuildError(f"profile evidence path must be repository-relative: {raw_path}")
        resolved = (resolved_root / declared).resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise ProfileBuildError(f"profile evidence path escapes the repository: {raw_path}") from error
        document = load_json(resolved)
        reviewed.append(
            {
                "evidence_id": evidence_id,
                "path": declared.as_posix(),
                "verification_status": raw_record.get("verification_status"),
                "review_mode": "logical_only",
                "json_root": "object",
                "top_level_fields": sorted(document),
            }
        )
    return reviewed


def validate_profile_capacity_semantics(profile: Mapping[str, Any]) -> None:
    """Reject legacy profiles that counted shared logical layer views."""

    # Test fixtures and synthetic unit-test profiles never drive a real GPU
    # run.  Calibration profiles are executable inputs and must be explicit.
    if profile.get("profile_status") in {"TEST_FIXTURE", "VERIFIED"}:
        return
    if profile.get("capacity_bytes_semantics") != CAPACITY_BYTES_SEMANTICS:
        raise ProfileBuildError(
            "controller requires capacity_bytes_semantics="
            f"{CAPACITY_BYTES_SEMANTICS!r}; legacy logical-view profiles are not executable"
        )


def extract_unique_option(args: Sequence[str], flag: str) -> str:
    """Extract one exact long-option value and reject duplicates or ambiguity."""

    values: list[str] = []
    prefix = f"{flag}="
    index = 0
    while index < len(args):
        item = str(args[index])
        if item == flag:
            if index + 1 >= len(args) or str(args[index + 1]).startswith("--"):
                raise ControllerError(f"missing value for {flag}")
            values.append(str(args[index + 1]))
            index += 2
            continue
        if item.startswith(prefix):
            values.append(item[len(prefix) :])
        index += 1
    if len(values) != 1:
        raise ControllerError(f"expected exactly one {flag}, found {len(values)}")
    return values[0]


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def validate_deployment_mapping(
    config: Mapping[str, Any],
    decision: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Prove that a policy decision maps to one executable vLLM command."""

    if decision.get("status") != "SELECTED" or not isinstance(decision.get("selected"), dict):
        raise ControllerError("controller requires a SELECTED policy decision")
    selected = decision["selected"]
    deployment = selected.get("deployment")
    if not isinstance(deployment, dict):
        raise ControllerError("selected decision is missing deployment mapping")
    allocation = deployment.get("allocation")
    if not isinstance(allocation, str) or allocation not in config.get("allocations", {}):
        raise ControllerError(f"selected allocation is not executable: {allocation!r}")

    controller_config = config.get("controller")
    if not isinstance(controller_config, dict):
        raise ControllerError("serving config is missing controller metadata")
    if controller_config.get("model_id") != request.get("model_id"):
        raise ControllerError("request model_id does not match serving config controller.model_id")

    allocation_args = [str(arg) for arg in config["allocations"][allocation].get("server_args", [])]
    precision_args = deployment.get("precision_args")
    if not isinstance(precision_args, list) or not all(isinstance(arg, str) for arg in precision_args):
        raise ControllerError("selected precision_args must be an array of strings")
    for flag in ("--kv-cache-dtype", "--mamba-ssm-cache-dtype"):
        expected = extract_unique_option(precision_args, flag)
        observed = extract_unique_option(allocation_args, flag)
        if observed != expected:
            raise ControllerError(f"runner mapping mismatch for {flag}: selected={expected!r} runner={observed!r}")

    server_args = [str(arg) for arg in config["server"].get("args", [])]
    observed_length = extract_unique_option(server_args, "--max-model-len")
    if not _same_number(observed_length, request.get("max_model_len")):
        raise ControllerError("request max_model_len does not match the frozen server command")
    observed_utilization = extract_unique_option(server_args, "--gpu-memory-utilization")
    memory_budget = request.get("memory_budget")
    if not isinstance(memory_budget, dict) or not _same_number(
        observed_utilization,
        memory_budget.get("gpu_memory_utilization"),
    ):
        raise ControllerError("request GPU memory utilization does not match the frozen server command")

    command = build_server_command(config, allocation, repo_root)
    return {
        "allocation": allocation,
        "precision_args": list(precision_args),
        "allocation_server_args": allocation_args,
        "server_command": command,
        "restart_required": deployment.get("restart_required") is True,
    }


def validate_execution_slice(
    config: Mapping[str, Any],
    phase_name: str,
    decision: Mapping[str, Any],
    request: Mapping[str, Any],
    seeds: list[int] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = decision["selected"]
    allocation = selected["deployment"]["allocation"]
    phase = resolve_phase(
        config,
        phase_name,
        allocation_filter=[allocation],
        workload_filter=[str(request["workload"])],
        seed_filter=seeds,
        rate_filter=[float(request["offered_rate_req_s"])],
    )
    if phase["allocations"] != [allocation]:
        raise ControllerError("phase did not resolve to exactly the selected allocation")
    rates = phase["workload_rates"].get(str(request["workload"]))
    if rates is None or len(rates) != 1 or not _same_number(rates[0], request["offered_rate_req_s"]):
        raise ControllerError("phase did not resolve to the exact requested workload and offered load")
    plan = build_sample_plan(config, phase)
    if not plan:
        raise ControllerError("controller execution slice is empty")
    return phase, plan


def collect_controller_result(attempt_dir: Path, expected_server_command: list[str]) -> dict[str, Any]:
    summary_path = attempt_dir / "summary.json"
    if not summary_path.is_file():
        raise ControllerError(f"runner summary is missing: {summary_path}")
    summary = load_json(summary_path)
    counts = summary.get("counts")
    if not isinstance(counts, dict) or set(counts) != {"completed_validated"}:
        raise ControllerError(f"runner summary is incomplete: {counts!r}")

    server_root = attempt_dir / "servers"
    contract_paths = sorted(server_root.glob("*/*/contract.json"))
    status_paths = sorted(server_root.glob("*/*/status.json"))
    if len(contract_paths) != 1 or len(status_paths) != 1:
        raise ControllerError(
            f"expected one deployment-epoch server session, found contracts={len(contract_paths)} "
            f"statuses={len(status_paths)}"
        )
    server_contract = load_json(contract_paths[0])
    server_status = load_json(status_paths[0])
    if server_contract.get("command") != expected_server_command:
        raise ControllerError("executed server command differs from the controller mapping")
    if server_status.get("status") != "stopped" or server_status.get("exception") is not None:
        raise ControllerError(f"server session did not stop cleanly: {server_status!r}")
    startup_duration = server_status.get("startup_duration_s")
    if not isinstance(startup_duration, (int, float)) or not math.isfinite(float(startup_duration)):
        raise ControllerError("server startup duration is missing or invalid")

    analysis_paths = sorted((attempt_dir / "samples").glob("*/analysis.json"))
    analyses = [load_json(path) for path in analysis_paths]
    if len(analyses) != int(counts["completed_validated"]):
        raise ControllerError("sample analysis count does not match the validated denominator")
    return {
        "summary": summary,
        "server_contract_path": str(contract_paths[0]),
        "server_contract_sha256": sha256_file(contract_paths[0]),
        "server_status_path": str(status_paths[0]),
        "server_status_sha256": sha256_file(status_paths[0]),
        "startup_duration_s": float(startup_duration),
        "sample_analyses": analyses,
    }


def parse_int_csv(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ControllerError("--seeds produced an empty seed set")
    if len(parsed) != len(set(parsed)):
        raise ControllerError("--seeds contains duplicates")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--serving-config", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--parent-attempt")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds")
    parser.add_argument(
        "--evidence-review-mode",
        choices=("hash_verified", "logical_only"),
        default="hash_verified",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile_path = args.profile.resolve()
    request_path = args.request.resolve()
    config_path = args.serving_config.resolve()
    output_root = args.output_root.resolve()
    attempt_dir = output_root / args.attempt_id
    if attempt_dir.exists():
        raise ControllerError(f"controller attempt already exists: {attempt_dir}; retries require a new attempt ID")
    attempt_dir.mkdir(parents=True, exist_ok=False)

    try:
        profile = load_json(profile_path)
        validate_profile_capacity_semantics(profile)
        request = load_json(request_path)
        config = load_config(config_path)
        repo_root = config_path.parents[2]
        if args.evidence_review_mode == "logical_only":
            evidence_verification = review_profile_evidence_logically(profile, repo_root)
        else:
            evidence_verification = verify_profile_evidence(profile, repo_root)
        root_git = get_git_state(repo_root, require_clean=bool(config["protocol"]["require_clean_git"]))
        seeds = parse_int_csv(args.seeds)
    except (ControllerError, ExperimentError, ProfileBuildError, OSError, ValueError, KeyError) as error:
        status = "INVALID_EVIDENCE" if isinstance(error, ProfileBuildError) else "PREFLIGHT_FAILED"
        write_json_with_hash(
            attempt_dir / "controller_failure.json",
            {
                "schema_version": 1,
                "status": status,
                "attempt_id": args.attempt_id,
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        return 3 if status == "INVALID_EVIDENCE" else 4

    selection_started = time.perf_counter_ns()
    try:
        decision = select_joint_precision(profile, request)
    except NoFeasibleCandidate as error:
        decision = error.report
        decision["decision_latency_ms"] = (time.perf_counter_ns() - selection_started) / 1_000_000
        write_json_with_hash(attempt_dir / "decision.json", decision)
        return 2
    except PolicyInputError as error:
        write_json_with_hash(
            attempt_dir / "controller_failure.json",
            {
                "schema_version": 1,
                "status": "INVALID_INPUT",
                "attempt_id": args.attempt_id,
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        return 3
    decision_latency_ms = (time.perf_counter_ns() - selection_started) / 1_000_000
    decision["decision_latency_ms"] = decision_latency_ms
    decision_sha = write_json_with_hash(attempt_dir / "decision.json", decision)

    if decision["profile_status"] == "TEST_FIXTURE" and not args.dry_run:
        write_json_with_hash(
            attempt_dir / "controller_failure.json",
            {
                "schema_version": 1,
                "status": "NON_EXECUTABLE_PROFILE",
                "attempt_id": args.attempt_id,
                "error": {
                    "type": "ControllerError",
                    "message": "TEST_FIXTURE profiles are restricted to --dry-run",
                },
            },
        )
        return 3

    try:
        mapping = validate_deployment_mapping(config, decision, decision["request"], repo_root=repo_root)
        phase, plan = validate_execution_slice(config, args.phase, decision, decision["request"], seeds)
    except (ControllerError, ExperimentError, OSError, ValueError, KeyError) as error:
        write_json_with_hash(
            attempt_dir / "controller_failure.json",
            {
                "schema_version": 1,
                "status": "PREFLIGHT_FAILED",
                "attempt_id": args.attempt_id,
                "decision_sha256": decision_sha,
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        return 4

    runner_output_root = attempt_dir / "runner"
    runner_attempt_dir = runner_output_root / args.attempt_id
    contract = {
        "schema_version": 1,
        "material_passport": {
            "origin_skill": "experiment-skill",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).date().isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "joint_precision_controller_contract_v1",
        },
        "attempt_id": args.attempt_id,
        "parent_attempt": args.parent_attempt,
        "dry_run": bool(args.dry_run),
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "evidence_verification": evidence_verification,
        "evidence_review": {
            "mode": args.evidence_review_mode,
            "hash_validation_performed": args.evidence_review_mode == "hash_verified",
            "sidecar_files_used_as_launch_gates": args.evidence_review_mode == "hash_verified",
        },
        "request_path": str(request_path),
        "request_sha256": sha256_file(request_path),
        "serving_config_path": str(config_path),
        "serving_config_sha256": sha256_file(config_path),
        "root_git": root_git,
        "phase": phase,
        "plan": plan,
        "decision_sha256": decision_sha,
        "deployment_mapping": mapping,
        "runner_attempt_dir": str(runner_attempt_dir),
        "runner_argv": [
            "--config",
            str(config_path),
            "--phase",
            args.phase,
            "--attempt-id",
            args.attempt_id,
            "--output-root",
            str(runner_output_root),
            *(["--parent-attempt", args.parent_attempt] if args.parent_attempt else []),
            "--allocations",
            mapping["allocation"],
            "--workloads",
            str(decision["request"]["workload"]),
            "--rates",
            f"{float(decision['request']['offered_rate_req_s']):g}",
            *(["--seeds", ",".join(str(seed) for seed in seeds)] if seeds is not None else []),
        ],
    }
    contract_sha = write_json_with_hash(attempt_dir / "controller_contract.json", contract)

    if args.dry_run:
        write_json_with_hash(
            attempt_dir / "controller_result.json",
            {
                "schema_version": 1,
                "status": "DRY_RUN_VALIDATED",
                "attempt_id": args.attempt_id,
                "controller_contract_sha256": contract_sha,
                "decision_latency_ms": decision_latency_ms,
                "selected_config_id": decision["selected"]["config_id"],
                "deployment_mapping": mapping,
                "sample_plan": plan,
            },
        )
        return 0

    try:
        returncode = run_steady_state(contract["runner_argv"])
        if returncode != 0:
            raise ControllerError(f"steady-state runner returned {returncode}")
        observed = collect_controller_result(runner_attempt_dir, mapping["server_command"])
        result = {
            "schema_version": 1,
            "material_passport": {
                "origin_skill": "experiment-skill",
                "origin_mode": "run",
                "origin_date": datetime.now(timezone.utc).date().isoformat(),
                "verification_status": "UNVERIFIED",
                "version_label": "joint_precision_controller_result_v1",
            },
            "status": "COMPLETED_UNVERIFIED",
            "attempt_id": args.attempt_id,
            "controller_contract_sha256": contract_sha,
            "decision_latency_ms": decision_latency_ms,
            "selected_config_id": decision["selected"]["config_id"],
            "previous_config_id": decision["request"].get("previous_config_id"),
            "transition_semantics": "deployment_epoch_cold_restart",
            "decision_plus_startup_s": decision_latency_ms / 1000.0 + observed["startup_duration_s"],
            **observed,
        }
        write_json_with_hash(attempt_dir / "controller_result.json", result)
        return 0
    except (ControllerError, ExperimentError, OSError, ValueError) as error:
        write_json_with_hash(
            attempt_dir / "controller_failure.json",
            {
                "schema_version": 1,
                "status": "FAILED",
                "attempt_id": args.attempt_id,
                "controller_contract_sha256": contract_sha,
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        return 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ControllerError, ExperimentError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(4)
