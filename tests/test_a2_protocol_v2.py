from scripts.analyze.verify_a2_protocol_v2 import evaluate_protocol


def test_protocol_accepts_tight_environment_sensitive_repeat() -> None:
    result = evaluate_protocol(
        {"legacy": 705_604, "uniform": 2_736_947, "packed": 2_280_448},
        {"legacy": 706_560, "uniform": 2_740_224, "packed": 2_283_520},
        {"packed_over_legacy": 3.231909, "packed_over_uniform": 0.833209},
        {"packed_over_legacy": 3.231884, "packed_over_uniform": 0.833333},
        capacity_tolerance=0.01,
        ratio_tolerance=0.001,
    )

    assert result["capacity_within_tolerance"] is True
    assert result["ratios_within_tolerance"] is True


def test_protocol_rejects_ratio_drift() -> None:
    result = evaluate_protocol(
        {"legacy": 100, "uniform": 400, "packed": 320},
        {"legacy": 100, "uniform": 400, "packed": 320},
        {"packed_over_legacy": 3.2, "packed_over_uniform": 0.8},
        {"packed_over_legacy": 3.0, "packed_over_uniform": 0.8},
        capacity_tolerance=0.01,
        ratio_tolerance=0.001,
    )

    assert result["capacity_within_tolerance"] is True
    assert result["ratios_within_tolerance"] is False
