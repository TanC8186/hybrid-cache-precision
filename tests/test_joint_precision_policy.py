from __future__ import annotations

from copy import deepcopy

import pytest

from kvcache.policy import (
    NoFeasibleCandidate,
    PolicyInputError,
    canonical_precision_args,
    select_joint_precision,
)

MODEL_ID = "Qwen/Qwen3.5-2B"
EVIDENCE_SHA = "a" * 64


def candidate(
    config_id: str,
    *,
    goodput_lcb: float,
    quality_low: float,
    cache_bytes: int = 80,
    allocator_slots: float = 64,
    ttft_ucb: float = 450,
    tpot_ucb: float = 150,
    kv_cache_dtype: str = "auto",
    state_cache_dtype: str = "float32",
) -> dict:
    return {
        "config_id": config_id,
        "kv_cache_dtype": kv_cache_dtype,
        "state_cache_dtype": state_cache_dtype,
        "deployment": {
            "engine": "vllm",
            "allocation": config_id,
            "precision_args": canonical_precision_args(kv_cache_dtype, state_cache_dtype),
            "restart_required": True,
        },
        "capacity_profiles": [
            {
                "model_id": MODEL_ID,
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.85,
                "cache_bytes": cache_bytes,
                "allocator_equivalent_sequence_slots": allocator_slots,
                "evidence_ids": ["capacity"],
            }
        ],
        "serving_profiles": [
            {
                "model_id": MODEL_ID,
                "max_model_len": 4096,
                "workload": "random60",
                "offered_rate_req_s": 40,
                "slo": {"p95_ttft_ms": 500, "p95_tpot_ms": 200},
                "slo_goodput_lcb_req_s": goodput_lcb,
                "p95_ttft_ucb_ms": ttft_ucb,
                "p95_tpot_ucb_ms": tpot_ucb,
                "n_independent_repeats": 3,
                "evidence_ids": ["serving"],
            }
        ],
        "quality_profiles": [
            {
                "model_id": MODEL_ID,
                "task": "gsm8k",
                "delta_ci95_low": quality_low,
                "delta_ci95_high": quality_low + 0.5,
                "inference_method": "intercept_only_ols_two_way_cluster_robust_cr1",
                "estimand": "mean paired accuracy difference over observed seed-item draws",
                "n_seed_item_draws": 1800,
                "n_item_clusters": 1017,
                "n_seed_clusters": 9,
                "cluster_degrees_of_freedom": 8,
                "evidence_ids": ["quality"],
            }
        ],
    }


def profile(candidates: list[dict]) -> dict:
    return {
        "schema_version": 2,
        "profile_status": "VERIFIED",
        "evidence": [
            {
                "evidence_id": evidence_id,
                "path": f"results/{evidence_id}.json",
                "sha256": EVIDENCE_SHA,
                "verification_status": "VERIFIED",
            }
            for evidence_id in ("capacity", "serving", "quality")
        ],
        "candidates": candidates,
    }


def request() -> dict:
    return {
        "model_id": MODEL_ID,
        "max_model_len": 4096,
        "workload": "random60",
        "offered_rate_req_s": 40,
        "memory_budget": {
            "gpu_memory_utilization": 0.85,
            "max_cache_bytes": 100,
        },
        "required_allocator_equivalent_sequence_slots": 50,
        "slo": {"p95_ttft_ms": 500, "p95_tpot_ms": 200},
        "quality_constraints": {"gsm8k": -1.5},
    }


def test_selector_chooses_highest_robust_goodput() -> None:
    calibration = profile(
        [
            candidate("full", goodput_lcb=32, quality_low=0),
            candidate(
                "joint",
                goodput_lcb=41,
                quality_low=-1,
                kv_cache_dtype="int4_per_token_head",
                state_cache_dtype="bfloat16",
            ),
        ]
    )

    report = select_joint_precision(calibration, request())

    assert report["status"] == "SELECTED"
    assert report["selected"]["config_id"] == "joint"
    assert report["selected"]["deployment"]["precision_args"] == [
        "--kv-cache-dtype",
        "int4_per_token_head",
        "--mamba-ssm-cache-dtype",
        "bfloat16",
    ]


def test_selector_uses_quality_lower_bound() -> None:
    calibration = profile(
        [
            candidate("full", goodput_lcb=32, quality_low=0),
            candidate("joint", goodput_lcb=60, quality_low=-2),
        ]
    )

    report = select_joint_precision(calibration, request())

    assert report["selected"]["config_id"] == "full"
    rejected = next(row for row in report["evaluations"] if row["config_id"] == "joint")
    assert rejected["rejection_reasons"] == ["quality_guardrail_violated:gsm8k"]


def test_selector_rejects_tail_latency_capacity_and_memory_violations() -> None:
    calibration = profile(
        [
            candidate("slow", goodput_lcb=80, quality_low=0, ttft_ucb=800),
            candidate("small", goodput_lcb=70, quality_low=0, allocator_slots=20),
            candidate("oversize", goodput_lcb=60, quality_low=0, cache_bytes=101),
            candidate("valid", goodput_lcb=30, quality_low=0),
        ]
    )

    report = select_joint_precision(calibration, request())

    assert report["selected"]["config_id"] == "valid"
    reasons = {row["config_id"]: row["rejection_reasons"] for row in report["evaluations"]}
    assert reasons["slow"] == ["ttft_slo_violated"]
    assert reasons["small"] == ["insufficient_allocator_equivalent_sequence_slots"]
    assert reasons["oversize"] == ["memory_budget_exceeded"]


def test_selector_fails_closed_on_unmeasured_serving_stratum() -> None:
    row = candidate("unknown", goodput_lcb=10, quality_low=0)
    row["serving_profiles"][0]["offered_rate_req_s"] = 35

    with pytest.raises(NoFeasibleCandidate) as error:
        select_joint_precision(profile([row]), request())

    assert error.value.report["status"] == "NO_FEASIBLE_CANDIDATE"
    assert error.value.report["evaluations"][0]["rejection_reasons"] == ["missing_serving_profile"]


def test_selector_does_not_interpolate_capacity_strata() -> None:
    row = candidate("unknown", goodput_lcb=10, quality_low=0)
    row["capacity_profiles"][0]["gpu_memory_utilization"] = 0.8

    with pytest.raises(NoFeasibleCandidate) as error:
        select_joint_precision(profile([row]), request())

    assert error.value.report["evaluations"][0]["rejection_reasons"] == ["missing_capacity_profile"]


def test_selector_breaks_equal_goodput_ties_by_quality_then_memory() -> None:
    calibration = profile(
        [
            candidate("more_memory", goodput_lcb=40, quality_low=-1.0, cache_bytes=90),
            candidate("less_memory", goodput_lcb=40, quality_low=-0.5, cache_bytes=70),
        ]
    )

    report = select_joint_precision(calibration, request())

    assert report["selected"]["config_id"] == "less_memory"


def test_selector_rejects_ambiguous_duplicate_evidence_rows() -> None:
    row = candidate("duplicate", goodput_lcb=40, quality_low=0)
    row["serving_profiles"].append(deepcopy(row["serving_profiles"][0]))

    with pytest.raises(PolicyInputError, match="ambiguous duplicate profile rows"):
        select_joint_precision(profile([row]), request())


def test_selector_rejects_noncanonical_runner_mapping() -> None:
    row = candidate("bad_mapping", goodput_lcb=40, quality_low=0)
    row["deployment"]["precision_args"][-1] = "float16"

    with pytest.raises(PolicyInputError, match="canonical dtype mapping"):
        select_joint_precision(profile([row]), request())


def test_selector_rejects_unknown_or_malformed_evidence() -> None:
    unknown = profile([candidate("unknown", goodput_lcb=40, quality_low=0)])
    unknown["candidates"][0]["capacity_profiles"][0]["evidence_ids"] = ["missing"]
    with pytest.raises(PolicyInputError, match="unknown evidence"):
        select_joint_precision(unknown, request())

    malformed = profile([candidate("malformed", goodput_lcb=40, quality_low=0)])
    malformed["evidence"][0]["sha256"] = "not-a-digest"
    with pytest.raises(PolicyInputError, match="lowercase SHA-256"):
        select_joint_precision(malformed, request())


def test_selector_accepts_explicitly_labeled_test_fixture_profile() -> None:
    fixture = profile([candidate("fixture", goodput_lcb=40, quality_low=0)])
    fixture["profile_status"] = "TEST_FIXTURE"
    for evidence in fixture["evidence"]:
        evidence["verification_status"] = "FIXTURE"

    report = select_joint_precision(fixture, request())

    assert report["profile_status"] == "TEST_FIXTURE"


def test_selector_decision_trace_exposes_constraints_and_quality_inference() -> None:
    report = select_joint_precision(
        profile([candidate("full", goodput_lcb=32, quality_low=0)]),
        request(),
    )

    evaluation = report["evaluations"][0]
    assert evaluation["objective_lcb_req_s"] == 32
    assert evaluation["constraint_checks"]["capacity"] == {
        "profile_found": True,
        "cache_bytes": 80,
        "max_cache_bytes": 100,
        "allocator_equivalent_sequence_slots": 64.0,
        "required_allocator_equivalent_sequence_slots": 50.0,
        "source_field": "allocator_equivalent_sequence_slots",
    }
    inference = evaluation["constraint_checks"]["quality"]["gsm8k"]["inference"]
    assert inference["method"] == "intercept_only_ols_two_way_cluster_robust_cr1"
    assert inference["n_seed_item_draws"] == 1800
    assert inference["n_item_clusters"] == 1017
    assert inference["n_seed_clusters"] == 9
    assert inference["cluster_degrees_of_freedom"] == 8


def test_selector_labels_historical_capacity_aliases() -> None:
    historical = candidate("historical", goodput_lcb=32, quality_low=0)
    capacity = historical["capacity_profiles"][0]
    capacity["max_concurrency"] = capacity.pop("allocator_equivalent_sequence_slots")
    historical_request = request()
    historical_request["required_concurrency"] = historical_request.pop(
        "required_allocator_equivalent_sequence_slots"
    )

    report = select_joint_precision(profile([historical]), historical_request)

    assert report["request"]["allocator_slot_requirement_input_field"] == (
        "required_concurrency (legacy alias)"
    )
    capacity_check = report["evaluations"][0]["constraint_checks"]["capacity"]
    assert capacity_check["source_field"] == "max_concurrency (legacy alias)"
