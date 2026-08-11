"""Fail-closed deployment-epoch selector for joint KV/state precision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class PolicyInputError(ValueError):
    """Raised when a policy input is incomplete or malformed."""


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
    memory_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "feasible": self.feasible,
            "rejection_reasons": list(self.rejection_reasons),
            "objective_lcb_req_s": self.objective_lcb_req_s,
            "quality_slack": self.quality_slack,
            "memory_bytes": self.memory_bytes,
        }


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyInputError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PolicyInputError(f"{field} must be finite")
    return result


def _positive_int(value: Any, field: str) -> int:
    number = _finite_number(value, field)
    if number <= 0 or not number.is_integer():
        raise PolicyInputError(f"{field} must be a positive integer")
    return int(number)


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyInputError(f"{field} must be an object")
    return value


def _validate_request(request: dict[str, Any]) -> dict[str, Any]:
    workload = request.get("workload")
    if not isinstance(workload, str) or not workload:
        raise PolicyInputError("request.workload must be a non-empty string")
    slo = _require_mapping(request.get("slo"), "request.slo")
    quality = _require_mapping(request.get("quality_constraints", {}), "request.quality_constraints")
    return {
        "workload": workload,
        "memory_budget_bytes": _positive_int(request.get("memory_budget_bytes"), "request.memory_budget_bytes"),
        "required_concurrency": _finite_number(
            request.get("required_concurrency"), "request.required_concurrency"
        ),
        "ttft_ms": _finite_number(slo.get("p95_ttft_ms"), "request.slo.p95_ttft_ms"),
        "tpot_ms": _finite_number(slo.get("p95_tpot_ms"), "request.slo.p95_tpot_ms"),
        "quality": {
            task: _finite_number(min_delta, f"request.quality_constraints.{task}")
            for task, min_delta in quality.items()
        },
    }


def _evaluate_candidate(candidate: dict[str, Any], request: dict[str, Any]) -> CandidateEvaluation:
    config_id = candidate.get("config_id")
    if not isinstance(config_id, str) or not config_id:
        raise PolicyInputError("candidate.config_id must be a non-empty string")
    memory_bytes = _positive_int(candidate.get("memory_bytes"), f"candidate.{config_id}.memory_bytes")
    max_concurrency = _finite_number(
        candidate.get("max_concurrency"), f"candidate.{config_id}.max_concurrency"
    )
    reasons: list[str] = []
    if memory_bytes > request["memory_budget_bytes"]:
        reasons.append("memory_budget_exceeded")
    if max_concurrency < request["required_concurrency"]:
        reasons.append("insufficient_concurrency")

    serving_map = _require_mapping(candidate.get("serving", {}), f"candidate.{config_id}.serving")
    serving = serving_map.get(request["workload"])
    objective: float | None = None
    if not isinstance(serving, dict):
        reasons.append("missing_workload_profile")
    else:
        objective = _finite_number(
            serving.get("slo_goodput_lcb_req_s"),
            f"candidate.{config_id}.serving.{request['workload']}.slo_goodput_lcb_req_s",
        )
        ttft_ucb = _finite_number(
            serving.get("p95_ttft_ucb_ms"),
            f"candidate.{config_id}.serving.{request['workload']}.p95_ttft_ucb_ms",
        )
        tpot_ucb = _finite_number(
            serving.get("p95_tpot_ucb_ms"),
            f"candidate.{config_id}.serving.{request['workload']}.p95_tpot_ucb_ms",
        )
        if ttft_ucb > request["ttft_ms"]:
            reasons.append("ttft_slo_violated")
        if tpot_ucb > request["tpot_ms"]:
            reasons.append("tpot_slo_violated")

    quality_map = _require_mapping(candidate.get("quality", {}), f"candidate.{config_id}.quality")
    slacks = []
    for task, min_delta in request["quality"].items():
        estimate = quality_map.get(task)
        if not isinstance(estimate, dict):
            reasons.append(f"missing_quality_profile:{task}")
            continue
        lower = _finite_number(estimate.get("delta_ci95_low"), f"candidate.{config_id}.quality.{task}")
        slacks.append(lower - min_delta)
        if lower < min_delta:
            reasons.append(f"quality_guardrail_violated:{task}")

    return CandidateEvaluation(
        config_id=config_id,
        feasible=not reasons,
        rejection_reasons=tuple(reasons),
        objective_lcb_req_s=objective,
        quality_slack=min(slacks) if slacks else math.inf,
        memory_bytes=memory_bytes,
    )


def select_joint_precision(profile: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Select the highest robust SLO-goodput candidate under frozen constraints.

    The selector uses lower confidence bounds for quality and goodput, and upper
    confidence bounds for tail latency. Missing evidence makes a candidate
    infeasible instead of triggering an extrapolation.
    """

    normalized_request = _validate_request(_require_mapping(request, "request"))
    candidates = profile.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PolicyInputError("profile.candidates must be a non-empty list")
    evaluations = [_evaluate_candidate(_require_mapping(c, "candidate"), normalized_request) for c in candidates]
    ids = [evaluation.config_id for evaluation in evaluations]
    if len(ids) != len(set(ids)):
        raise PolicyInputError("candidate config_id values must be unique")

    feasible = [evaluation for evaluation in evaluations if evaluation.feasible]
    report = {
        "schema_version": 1,
        "policy": "joint_precision_deployment_epoch_v1",
        "objective": "maximize robust SLO-goodput lower confidence bound",
        "request": normalized_request,
        "evaluations": [evaluation.as_dict() for evaluation in evaluations],
        "selected": None,
    }
    if not feasible:
        report["status"] = "NO_FEASIBLE_CANDIDATE"
        raise NoFeasibleCandidate(report)

    selected = sorted(
        feasible,
        key=lambda item: (
            -float(item.objective_lcb_req_s),
            -float(item.quality_slack),
            item.memory_bytes,
            item.config_id,
        ),
    )[0]
    selected_profile = next(candidate for candidate in candidates if candidate["config_id"] == selected.config_id)
    report["status"] = "SELECTED"
    report["selected"] = {
        "config_id": selected.config_id,
        "kv_dtype": selected_profile.get("kv_dtype"),
        "state_dtype": selected_profile.get("state_dtype"),
        "objective_lcb_req_s": selected.objective_lcb_req_s,
        "quality_slack": selected.quality_slack,
        "memory_bytes": selected.memory_bytes,
        "max_concurrency": selected_profile.get("max_concurrency"),
    }
    return report
