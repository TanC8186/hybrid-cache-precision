from scripts.analyze.verify_a2_comparative_protocol_v3_pilot import (
    classify_pilot,
    full_threshold_bracket_report,
)


def make_sample(allocation: str, workload: str, rate: float, thresholds: list[int]) -> dict:
    return {
        "allocation": allocation,
        "workload": workload,
        "offered_rate": rate,
        "sustainable_ttft_thresholds_ms": thresholds,
    }


def test_random_component_passes_only_when_every_gate_passes() -> None:
    verdict = classify_pilot(
        component="random",
        requests_passed=True,
        bracketed=True,
        servers_clean=True,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_RANDOM_PILOT_COMPONENT_PASSED"


def test_suite_fails_closed() -> None:
    verdict = classify_pilot(
        component="suite",
        requests_passed=True,
        bracketed=False,
        servers_clean=True,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_COMPARATIVE_PILOT_REVIEW_REQUIRED"


def test_pilot_bracket_requires_every_threshold_and_allocation() -> None:
    thresholds = [250, 500, 1000, 2000, 3000]
    samples = []
    for allocation in ("fp16", "int4", "packed_per_layer"):
        samples.append(make_sample(allocation, "random", 30.0, thresholds))
        samples.append(make_sample(allocation, "random", 50.0, []))

    _, bracketed = full_threshold_bracket_report(samples, "random")

    assert bracketed is True
