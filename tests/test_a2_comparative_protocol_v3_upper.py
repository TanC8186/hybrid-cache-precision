from scripts.analyze.verify_a2_comparative_protocol_v3_upper import (
    classify_linked_protocol_v3_chain,
    full_threshold_bracket_report,
)


def make_sample(allocation: str, rate: float, thresholds: list[int]) -> dict:
    return {
        "allocation": allocation,
        "workload": "sharegpt",
        "offered_rate": rate,
        "sustainable_ttft_thresholds_ms": thresholds,
    }


def test_linked_chain_requires_every_gate() -> None:
    verdict = classify_linked_protocol_v3_chain(
        fd_mvex_passed=True,
        lower_mvex_integrity_passed=True,
        upper_mvex_integrity_passed=True,
        full_threshold_bracketed=True,
        servers_clean=True,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_LINKED_BRACKET_MVEX_PASSED"


def test_linked_chain_fails_closed() -> None:
    verdict = classify_linked_protocol_v3_chain(
        fd_mvex_passed=True,
        lower_mvex_integrity_passed=True,
        upper_mvex_integrity_passed=True,
        full_threshold_bracketed=False,
        servers_clean=True,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_LINKED_BRACKET_MVEX_REVIEW_REQUIRED"


def test_full_threshold_bracket_requires_rate30_and_rate40_for_every_allocation() -> None:
    thresholds = [250, 500, 1000, 2000, 3000]
    lower = [
        make_sample(allocation, 30.0, thresholds)
        for allocation in ("fp16", "int4", "packed_per_layer")
    ]
    upper = [
        make_sample(allocation, 40.0, [])
        for allocation in ("fp16", "int4", "packed_per_layer")
    ]

    report, bracketed = full_threshold_bracket_report(lower, upper)

    assert bracketed is True
    assert all(
        item["bracketed"]
        for allocation in report.values()
        for item in allocation.values()
    )


def test_full_threshold_bracket_rejects_one_sustainable_upper_threshold() -> None:
    thresholds = [250, 500, 1000, 2000, 3000]
    lower = [
        make_sample(allocation, 30.0, thresholds)
        for allocation in ("fp16", "int4", "packed_per_layer")
    ]
    upper = [
        make_sample("fp16", 40.0, [3000]),
        make_sample("int4", 40.0, []),
        make_sample("packed_per_layer", 40.0, []),
    ]

    _, bracketed = full_threshold_bracket_report(lower, upper)

    assert bracketed is False
