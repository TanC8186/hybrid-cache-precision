"""Fail-closed deployment-epoch selector for joint KV/state precision."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

PROFILE_SCHEMA_VERSION = 2
POLICY_NAME = "joint_precision_deployment_epoch_v2"
SUPPORTED_KV_CACHE_DTYPES = {"auto", "int4_per_token_head"}
SUPPORTED_STATE_CACHE_DTYPES = {"float32", "bfloat16", "float16"}
EVIDENCE_STATUSES = {"ANALYZED", "FIXTURE", "VERIFIED"}
PROFILE_STATUSES = {"CALIBRATION", "TEST_FIXTURE", "VERIFIED"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PolicyInputError(ValueError):
    """Raised when a policy input is incomplete, ambiguous, or malformed."""


class NoFeasibleCandidate(RuntimeError):
    """Raised when every candidate violates at least one frozen constraint."""

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__("no candidate satisfies the frozen deployment constraints")
        self.report = report


@dataclass(frozen=True)
class CandidateEvaluation:
    config_id: str
    feasible: bool
    rejection_reasons: tuple[str, ...]
    objective_lcb_req_s: float | None
    quality_slack: float | None
    cache_bytes: int | None
    max_concurrency: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "feasible": self.feasible,
            "rejection_reasons": list(self.rejection_reasons),
            "objective_lcb_req_s": self.objective_lcb_req_s,
            "quality_slack": self.quality_slack,
            "cache_bytes": self.cache_bytes,
            "max_concurrency": self.max_concurrency,
        }


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyInputError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PolicyInputError(f"{field} must be finite")
    return result


def _positive_number(value: Any, field: str) -> float:
    result = _finite_number(value, field)
    if result <= 0:
        raise PolicyInputError(f"{field} must be positive")
    return result


def _positive_int(value: Any, field: str) -> int:
    number = _positive_number(value, field)
    if not number.is_integer():
        raise PolicyInputError(f"{field} must be a positive integer")
    return int(number)


def _fraction(value: Any, field: str) -> float:
    number = _positive_number(value, field)
    if number > 1:
        raise PolicyInputError(f"{field} must be in (0, 1]")
    return number


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyInputError(f"{field} must be a non-empty string")
    return value


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyInputError(f"{field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise PolicyInputError(f"{field} must be an array")
    return value


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def canonical_precision_args(kv_cache_dtype: str, state_cache_dtype: str) -> list[str]:
    """Return the only accepted vLLM precision argument mapping."""

    if kv_cache_dtype not in SUPPORTED_KV_CACHE_DTYPES:
        raise PolicyInputError(f"unsupported kv_cache_dtype: {kv_cache_dtype!r}")
    if state_cache_dtype not in SUPPORTED_STATE_CACHE_DTYPES:
        raise PolicyInputError(f"unsupported state_cache_dtype: {state_cache_dtype!r}")
    return [
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--mamba-ssm-cache-dtype",
        state_cache_dtype,
    ]


def _validate_request(request: dict[str, Any]) -> dict[str, Any]:
    memory = _require_mapping(request.get("memory_budget"), "request.memory_budget")
    slo = _require_mapping(request.get("slo"), "request.slo")
    quality = _require_mapping(request.get("quality_constraints", {}), "request.quality_constraints")
    normalized_quality: dict[str, float] = {}
    for task, minimum_delta in quality.items():
        task_name = _nonempty_string(task, "request.quality_constraints task")
        normalized_quality[task_name] = _finite_number(
            minimum_delta,
            f"request.quality_constraints.{task_name}",
        )
    normalized = {
        "model_id": _nonempty_string(request.get("model_id"), "request.model_id"),
        "max_model_len": _positive_int(request.get("max_model_len"), "request.max_model_len"),
        "workload": _nonempty_string(request.get("workload"), "request.workload"),
        "offered_rate_req_s": _positive_number(
            request.get("offered_rate_req_s"),
            "request.offered_rate_req_s",
        ),
        "required_concurrency": _positive_number(
            request.get("required_concurrency"),
            "request.required_concurrency",
        ),
        "memory_budget": {
            "gpu_memory_utilization": _fraction(
                memory.get("gpu_memory_utilization"),
                "request.memory_budget.gpu_memory_utilization",
            ),
            "max_cache_bytes": _positive_int(
                memory.get("max_cache_bytes"),
                "request.memory_budget.max_cache_bytes",
            ),
        },
        "slo": {
            "p95_ttft_ms": _positive_number(slo.get("p95_ttft_ms"), "request.slo.p95_ttft_ms"),
            "p95_tpot_ms": _positive_number(slo.get("p95_tpot_ms"), "request.slo.p95_tpot_ms"),
        },
        "quality_constraints": normalized_quality,
    }
    if "deployment_epoch_id" in request:
        normalized["deployment_epoch_id"] = _nonempty_string(
            request["deployment_epoch_id"],
            "request.deployment_epoch_id",
        )
    if "previous_config_id" in request and request["previous_config_id"] is not None:
        normalized["previous_config_id"] = _nonempty_string(
            request["previous_config_id"],
            "request.previous_config_id",
        )
    return normalized


def _validate_evidence(profile: dict[str, Any]) -> set[str]:
    records = _require_list(profile.get("evidence"), "profile.evidence")
    if not records:
        raise PolicyInputError("profile.evidence must be non-empty")
    evidence_ids: set[str] = set()
    for index, raw in enumerate(records):
        record = _require_mapping(raw, f"profile.evidence[{index}]")
        evidence_id = _nonempty_string(record.get("evidence_id"), f"profile.evidence[{index}].evidence_id")
        if evidence_id in evidence_ids:
            raise PolicyInputError(f"duplicate evidence_id: {evidence_id}")
        evidence_ids.add(evidence_id)
        _nonempty_string(record.get("path"), f"profile.evidence.{evidence_id}.path")
        digest = _nonempty_string(record.get("sha256"), f"profile.evidence.{evidence_id}.sha256")
        if SHA256_RE.fullmatch(digest) is None:
            raise PolicyInputError(f"profile.evidence.{evidence_id}.sha256 must be lowercase SHA-256")
        status = _nonempty_string(
            record.get("verification_status"),
            f"profile.evidence.{evidence_id}.verification_status",
        )
        if status not in EVIDENCE_STATUSES:
            raise PolicyInputError(
                f"profile.evidence.{evidence_id}.verification_status must be one of {sorted(EVIDENCE_STATUSES)}"
            )
    return evidence_ids


def _validate_evidence_ids(value: Any, field: str, known_ids: set[str]) -> list[str]:
    raw_ids = _require_list(value, field)
    if not raw_ids:
        raise PolicyInputError(f"{field} must be non-empty")
    ids = [_nonempty_string(item, field) for item in raw_ids]
    if len(ids) != len(set(ids)):
        raise PolicyInputError(f"{field} contains duplicates")
    unknown = sorted(set(ids) - known_ids)
    if unknown:
        raise PolicyInputError(f"{field} references unknown evidence: {unknown}")
    return ids


def _validate_capacity_profiles(value: Any, field: str, evidence_ids: set[str]) -> list[dict[str, Any]]:
    rows = _require_list(value, field)
    if not rows:
        raise PolicyInputError(f"{field} must be non-empty")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row_field = f"{field}[{index}]"
        row = _require_mapping(raw, row_field)
        normalized.append(
            {
                "model_id": _nonempty_string(row.get("model_id"), f"{row_field}.model_id"),
                "max_model_len": _positive_int(row.get("max_model_len"), f"{row_field}.max_model_len"),
                "gpu_memory_utilization": _fraction(
                    row.get("gpu_memory_utilization"),
                    f"{row_field}.gpu_memory_utilization",
                ),
                "cache_bytes": _positive_int(row.get("cache_bytes"), f"{row_field}.cache_bytes"),
                "max_concurrency": _positive_number(
                    row.get("max_concurrency"),
                    f"{row_field}.max_concurrency",
                ),
                "evidence_ids": _validate_evidence_ids(
                    row.get("evidence_ids"),
                    f"{row_field}.evidence_ids",
                    evidence_ids,
                ),
            }
        )
    return normalized


def _validate_serving_profiles(value: Any, field: str, evidence_ids: set[str]) -> list[dict[str, Any]]:
    rows = _require_list(value, field)
    if not rows:
        raise PolicyInputError(f"{field} must be non-empty")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row_field = f"{field}[{index}]"
        row = _require_mapping(raw, row_field)
        slo = _require_mapping(row.get("slo"), f"{row_field}.slo")
        normalized.append(
            {
                "model_id": _nonempty_string(row.get("model_id"), f"{row_field}.model_id"),
                "max_model_len": _positive_int(row.get("max_model_len"), f"{row_field}.max_model_len"),
                "workload": _nonempty_string(row.get("workload"), f"{row_field}.workload"),
                "offered_rate_req_s": _positive_number(
                    row.get("offered_rate_req_s"),
                    f"{row_field}.offered_rate_req_s",
                ),
                "slo": {
                    "p95_ttft_ms": _positive_number(
                        slo.get("p95_ttft_ms"),
                        f"{row_field}.slo.p95_ttft_ms",
                    ),
                    "p95_tpot_ms": _positive_number(
                        slo.get("p95_tpot_ms"),
                        f"{row_field}.slo.p95_tpot_ms",
                    ),
                },
                "slo_goodput_lcb_req_s": _positive_number(
                    row.get("slo_goodput_lcb_req_s"),
                    f"{row_field}.slo_goodput_lcb_req_s",
                ),
                "p95_ttft_ucb_ms": _positive_number(
                    row.get("p95_ttft_ucb_ms"),
                    f"{row_field}.p95_ttft_ucb_ms",
                ),
                "p95_tpot_ucb_ms": _positive_number(
                    row.get("p95_tpot_ucb_ms"),
                    f"{row_field}.p95_tpot_ucb_ms",
                ),
                "n_independent_repeats": _positive_int(
                    row.get("n_independent_repeats"),
                    f"{row_field}.n_independent_repeats",
                ),
                "evidence_ids": _validate_evidence_ids(
                    row.get("evidence_ids"),
                    f"{row_field}.evidence_ids",
                    evidence_ids,
                ),
            }
        )
    return normalized


def _validate_quality_profiles(value: Any, field: str, evidence_ids: set[str]) -> list[dict[str, Any]]:
    rows = _require_list(value, field)
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row_field = f"{field}[{index}]"
        row = _require_mapping(raw, row_field)
        low = _finite_number(row.get("delta_ci95_low"), f"{row_field}.delta_ci95_low")
        high = _finite_number(row.get("delta_ci95_high"), f"{row_field}.delta_ci95_high")
        if low > high:
            raise PolicyInputError(f"{row_field} has delta_ci95_low > delta_ci95_high")
        normalized.append(
            {
                "model_id": _nonempty_string(row.get("model_id"), f"{row_field}.model_id"),
                "task": _nonempty_string(row.get("task"), f"{row_field}.task"),
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "n_independent_repeats": _positive_int(
                    row.get("n_independent_repeats"),
                    f"{row_field}.n_independent_repeats",
                ),
                "evidence_ids": _validate_evidence_ids(
                    row.get("evidence_ids"),
                    f"{row_field}.evidence_ids",
                    evidence_ids,
                ),
            }
        )
    return normalized


def _validate_candidate(raw: Any, index: int, evidence_ids: set[str]) -> dict[str, Any]:
    field = f"profile.candidates[{index}]"
    candidate = _require_mapping(raw, field)
    config_id = _nonempty_string(candidate.get("config_id"), f"{field}.config_id")
    kv_dtype = _nonempty_string(candidate.get("kv_cache_dtype"), f"{field}.kv_cache_dtype")
    state_dtype = _nonempty_string(candidate.get("state_cache_dtype"), f"{field}.state_cache_dtype")
    expected_args = canonical_precision_args(kv_dtype, state_dtype)
    deployment = _require_mapping(candidate.get("deployment"), f"candidate.{config_id}.deployment")
    if deployment.get("engine") != "vllm":
        raise PolicyInputError(f"candidate.{config_id}.deployment.engine must be 'vllm'")
    allocation = _nonempty_string(
        deployment.get("allocation"),
        f"candidate.{config_id}.deployment.allocation",
    )
    precision_args = _require_list(
        deployment.get("precision_args"),
        f"candidate.{config_id}.deployment.precision_args",
    )
    if not all(isinstance(item, str) for item in precision_args):
        raise PolicyInputError(f"candidate.{config_id}.deployment.precision_args must contain strings")
    if precision_args != expected_args:
        raise PolicyInputError(
            f"candidate.{config_id}.deployment.precision_args does not match the canonical dtype mapping"
        )
    if deployment.get("restart_required") is not True:
        raise PolicyInputError(f"candidate.{config_id}.deployment.restart_required must be true")
    return {
        "config_id": config_id,
        "kv_cache_dtype": kv_dtype,
        "state_cache_dtype": state_dtype,
        "deployment": {
            "engine": "vllm",
            "allocation": allocation,
            "precision_args": list(precision_args),
            "restart_required": True,
        },
        "capacity_profiles": _validate_capacity_profiles(
            candidate.get("capacity_profiles"),
            f"candidate.{config_id}.capacity_profiles",
            evidence_ids,
        ),
        "serving_profiles": _validate_serving_profiles(
            candidate.get("serving_profiles"),
            f"candidate.{config_id}.serving_profiles",
            evidence_ids,
        ),
        "quality_profiles": _validate_quality_profiles(
            candidate.get("quality_profiles", []),
            f"candidate.{config_id}.quality_profiles",
            evidence_ids,
        ),
    }


def _validate_profile(profile: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise PolicyInputError(f"profile.schema_version must be {PROFILE_SCHEMA_VERSION}")
    profile_status = _nonempty_string(profile.get("profile_status"), "profile.profile_status")
    if profile_status not in PROFILE_STATUSES:
        raise PolicyInputError(f"profile.profile_status must be one of {sorted(PROFILE_STATUSES)}")
    evidence_ids = _validate_evidence(profile)
    raw_candidates = _require_list(profile.get("candidates"), "profile.candidates")
    if not raw_candidates:
        raise PolicyInputError("profile.candidates must be non-empty")
    candidates = [_validate_candidate(raw, index, evidence_ids) for index, raw in enumerate(raw_candidates)]
    ids = [candidate["config_id"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise PolicyInputError("candidate config_id values must be unique")
    allocations = [candidate["deployment"]["allocation"] for candidate in candidates]
    if len(allocations) != len(set(allocations)):
        raise PolicyInputError("candidate deployment allocation values must be unique")
    return profile_status, candidates


def validate_joint_precision_profile(profile: dict[str, Any]) -> None:
    """Validate a profile without evaluating a deployment request."""

    _validate_profile(_require_mapping(profile, "profile"))


def _find_unique(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    field: str,
) -> dict[str, Any] | None:
    matches = [row for row in rows if predicate(row)]
    if len(matches) > 1:
        raise PolicyInputError(f"ambiguous duplicate profile rows for {field}")
    return matches[0] if matches else None


def _evaluate_candidate(candidate: dict[str, Any], request: dict[str, Any]) -> CandidateEvaluation:
    config_id = candidate["config_id"]
    reasons: list[str] = []
    capacity = _find_unique(
        candidate["capacity_profiles"],
        lambda row: (
            row["model_id"] == request["model_id"]
            and row["max_model_len"] == request["max_model_len"]
            and _same_number(
                row["gpu_memory_utilization"],
                request["memory_budget"]["gpu_memory_utilization"],
            )
        ),
        f"candidate.{config_id}.capacity",
    )
    cache_bytes: int | None = None
    max_concurrency: float | None = None
    if capacity is None:
        reasons.append("missing_capacity_profile")
    else:
        cache_bytes = capacity["cache_bytes"]
        max_concurrency = capacity["max_concurrency"]
        if cache_bytes > request["memory_budget"]["max_cache_bytes"]:
            reasons.append("memory_budget_exceeded")
        if max_concurrency < request["required_concurrency"]:
            reasons.append("insufficient_concurrency")

    serving = _find_unique(
        candidate["serving_profiles"],
        lambda row: (
            row["model_id"] == request["model_id"]
            and row["max_model_len"] == request["max_model_len"]
            and row["workload"] == request["workload"]
            and _same_number(row["offered_rate_req_s"], request["offered_rate_req_s"])
            and _same_number(row["slo"]["p95_ttft_ms"], request["slo"]["p95_ttft_ms"])
            and _same_number(row["slo"]["p95_tpot_ms"], request["slo"]["p95_tpot_ms"])
        ),
        f"candidate.{config_id}.serving",
    )
    objective: float | None = None
    if serving is None:
        reasons.append("missing_serving_profile")
    else:
        objective = serving["slo_goodput_lcb_req_s"]
        if serving["p95_ttft_ucb_ms"] > request["slo"]["p95_ttft_ms"]:
            reasons.append("ttft_slo_violated")
        if serving["p95_tpot_ucb_ms"] > request["slo"]["p95_tpot_ms"]:
            reasons.append("tpot_slo_violated")

    slacks: list[float] = []
    for task, minimum_delta in request["quality_constraints"].items():
        quality = _find_unique(
            candidate["quality_profiles"],
            lambda row, task=task: row["model_id"] == request["model_id"] and row["task"] == task,
            f"candidate.{config_id}.quality.{task}",
        )
        if quality is None:
            reasons.append(f"missing_quality_profile:{task}")
            continue
        slack = quality["delta_ci95_low"] - minimum_delta
        slacks.append(slack)
        if slack < 0:
            reasons.append(f"quality_guardrail_violated:{task}")

    return CandidateEvaluation(
        config_id=config_id,
        feasible=not reasons,
        rejection_reasons=tuple(reasons),
        objective_lcb_req_s=objective,
        quality_slack=min(slacks) if slacks else None,
        cache_bytes=cache_bytes,
        max_concurrency=max_concurrency,
    )


def select_joint_precision(profile: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Select the highest robust SLO-goodput candidate for an observed stratum.

    Capacity, serving, and quality evidence must match the requested model,
    context, memory budget, workload, offered load, and SLO exactly. The policy
    does not interpolate or extrapolate across unmeasured strata.
    """

    normalized_profile = _require_mapping(profile, "profile")
    normalized_request = _validate_request(_require_mapping(request, "request"))
    profile_status, candidates = _validate_profile(normalized_profile)
    evaluations = [_evaluate_candidate(candidate, normalized_request) for candidate in candidates]
    feasible = [evaluation for evaluation in evaluations if evaluation.feasible]
    report = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "policy": POLICY_NAME,
        "profile_status": profile_status,
        "objective": "maximize robust SLO-goodput lower confidence bound",
        "request": normalized_request,
        "evaluations": [evaluation.as_dict() for evaluation in evaluations],
        "selected": None,
    }
    if not feasible:
        report["status"] = "NO_FEASIBLE_CANDIDATE"
        raise NoFeasibleCandidate(report)

    def selection_key(item: CandidateEvaluation) -> tuple[float, float, int, float, str]:
        assert item.objective_lcb_req_s is not None
        assert item.cache_bytes is not None
        assert item.max_concurrency is not None
        quality_slack = item.quality_slack if item.quality_slack is not None else math.inf
        return (
            -item.objective_lcb_req_s,
            -quality_slack,
            item.cache_bytes,
            -item.max_concurrency,
            item.config_id,
        )

    selected = min(feasible, key=selection_key)
    selected_profile = next(candidate for candidate in candidates if candidate["config_id"] == selected.config_id)
    report["status"] = "SELECTED"
    report["selected"] = {
        "config_id": selected.config_id,
        "kv_cache_dtype": selected_profile["kv_cache_dtype"],
        "state_cache_dtype": selected_profile["state_cache_dtype"],
        "objective_lcb_req_s": selected.objective_lcb_req_s,
        "quality_slack": selected.quality_slack,
        "cache_bytes": selected.cache_bytes,
        "max_concurrency": selected.max_concurrency,
        "deployment": selected_profile["deployment"],
    }
    return report
