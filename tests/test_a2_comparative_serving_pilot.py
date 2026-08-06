from scripts.analyze.verify_a2_comparative_serving_pilot import (
    classify_comparative_pilot,
    detect_runtime_faults,
    sharegpt_bracket_report,
)


def test_comparative_pilot_pass_requires_every_gate() -> None:
    verdict = classify_comparative_pilot(
        mvex_passed=True,
        pilot_requests_passed=True,
        servers_clean=True,
        graph_mode_proven=True,
        sharegpt_bracketed=True,
        fd_limit_failure_proven=False,
    )

    assert verdict == "PILOT_PASSED_PREDECLARED_CRITERIA"


def test_comparative_pilot_classifies_observed_failure() -> None:
    verdict = classify_comparative_pilot(
        mvex_passed=True,
        pilot_requests_passed=False,
        servers_clean=True,
        graph_mode_proven=True,
        sharegpt_bracketed=False,
        fd_limit_failure_proven=True,
    )

    assert verdict == "PILOT_FAILED_CLIENT_FD_LIMIT_AND_SHAREGPT_WINDOW_BRACKETING"


def test_comparative_pilot_fails_closed_for_unknown_combination() -> None:
    verdict = classify_comparative_pilot(
        mvex_passed=True,
        pilot_requests_passed=False,
        servers_clean=False,
        graph_mode_proven=True,
        sharegpt_bracketed=False,
        fd_limit_failure_proven=True,
    )

    assert verdict == "COMPARATIVE_PILOT_INTEGRITY_REVIEW_REQUIRED"


def test_sharegpt_bracket_requires_each_allocation() -> None:
    samples = []
    for allocation in ("fp16", "int4", "packed_per_layer"):
        samples.extend(
            [
                {
                    "allocation": allocation,
                    "workload": "sharegpt",
                    "sustainable_ttft_thresholds_ms": [1000],
                },
                {
                    "allocation": allocation,
                    "workload": "sharegpt",
                    "sustainable_ttft_thresholds_ms": [],
                },
            ]
        )

    report, bracketed = sharegpt_bracket_report(samples)

    assert bracketed is True
    assert report["fp16"]["1000"] == {
        "has_sustainable": True,
        "has_unsustainable": True,
    }


def test_sharegpt_all_unsustainable_is_not_a_bracket() -> None:
    samples = [
        {
            "allocation": allocation,
            "workload": "sharegpt",
            "sustainable_ttft_thresholds_ms": [],
        }
        for allocation in ("fp16", "int4", "packed_per_layer")
    ]

    _, bracketed = sharegpt_bracket_report(samples)

    assert bracketed is False


def test_shutdown_noise_is_not_a_runtime_fault() -> None:
    faults = detect_runtime_faults("EngineDeadError during normal SIGTERM shutdown")

    assert not any(faults.values())
