import pytest

from kvcache.policy import NoFeasibleCandidate, select_joint_precision


def candidate(
    config_id: str,
    *,
    goodput_lcb: float,
    quality_low: float,
    memory_bytes: int = 80,
    max_concurrency: float = 64,
    ttft_ucb: float = 450,
    tpot_ucb: float = 150,
) -> dict:
    return {
        "config_id": config_id,
        "kv_dtype": "int4" if "int4" in config_id else "fp16",
        "state_dtype": "bfloat16" if "bf16" in config_id else "float32",
        "memory_bytes": memory_bytes,
        "max_concurrency": max_concurrency,
        "quality": {"gsm8k": {"delta_ci95_low": quality_low}},
        "serving": {
            "random60": {
                "slo_goodput_lcb_req_s": goodput_lcb,
                "p95_ttft_ucb_ms": ttft_ucb,
                "p95_tpot_ucb_ms": tpot_ucb,
            }
        },
    }


def request() -> dict:
    return {
        "workload": "random60",
        "memory_budget_bytes": 100,
        "required_concurrency": 50,
        "slo": {"p95_ttft_ms": 500, "p95_tpot_ms": 200},
        "quality_constraints": {"gsm8k": -1.5},
    }


def test_selector_chooses_highest_robust_goodput() -> None:
    profile = {
        "candidates": [
            candidate("fp16_fp32", goodput_lcb=32, quality_low=0),
            candidate("int4_bf16", goodput_lcb=41, quality_low=-1),
        ]
    }

    report = select_joint_precision(profile, request())

    assert report["status"] == "SELECTED"
    assert report["selected"]["config_id"] == "int4_bf16"


def test_selector_uses_quality_lower_bound() -> None:
    profile = {
        "candidates": [
            candidate("fp16_fp32", goodput_lcb=32, quality_low=0),
            candidate("int4_bf16", goodput_lcb=60, quality_low=-2),
        ]
    }

    report = select_joint_precision(profile, request())

    assert report["selected"]["config_id"] == "fp16_fp32"
    rejected = next(row for row in report["evaluations"] if row["config_id"] == "int4_bf16")
    assert rejected["rejection_reasons"] == ["quality_guardrail_violated:gsm8k"]


def test_selector_rejects_tail_latency_and_capacity_violations() -> None:
    profile = {
        "candidates": [
            candidate("slow", goodput_lcb=80, quality_low=0, ttft_ucb=800),
            candidate("small", goodput_lcb=70, quality_low=0, max_concurrency=20),
            candidate("valid", goodput_lcb=30, quality_low=0),
        ]
    }

    report = select_joint_precision(profile, request())

    assert report["selected"]["config_id"] == "valid"


def test_selector_fails_closed_on_missing_workload_evidence() -> None:
    row = candidate("unknown", goodput_lcb=10, quality_low=0)
    row["serving"] = {}

    with pytest.raises(NoFeasibleCandidate) as error:
        select_joint_precision({"candidates": [row]}, request())

    assert error.value.report["status"] == "NO_FEASIBLE_CANDIDATE"
    assert error.value.report["evaluations"][0]["rejection_reasons"] == ["missing_workload_profile"]


def test_selector_breaks_equal_goodput_ties_by_quality_then_memory() -> None:
    profile = {
        "candidates": [
            candidate("more_memory", goodput_lcb=40, quality_low=-1.0, memory_bytes=90),
            candidate("less_memory", goodput_lcb=40, quality_low=-0.5, memory_bytes=70),
        ]
    }

    report = select_joint_precision(profile, request())

    assert report["selected"]["config_id"] == "less_memory"
