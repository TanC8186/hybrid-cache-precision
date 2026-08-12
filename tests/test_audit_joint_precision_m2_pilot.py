from __future__ import annotations

import math

import pytest

from scripts.analyze.audit_joint_precision_m2_pilot import (
    AuditError,
    check_result_accounting,
    command_value,
    fallacy_scan,
    student_t_summary,
)


def test_student_t_summary_uses_seed_level_df2_interval() -> None:
    summary = student_t_summary([1.0, 2.0, 3.0])

    assert summary["n"] == 3
    assert summary["mean"] == 2.0
    assert summary["sample_sd"] == 1.0
    assert math.isclose(summary["ci95_low"], -0.48413771171954556)
    assert math.isclose(summary["ci95_high"], 4.484137711719546)


def test_result_accounting_fails_closed_on_reduced_denominator() -> None:
    result = {
        "completed": 2,
        "failed": 0,
        "ttfts": [0.1, 0.2],
        "itls": [[], []],
        "input_lens": [1, 1],
        "output_lens": [1, 1],
        "start_times": [0.0, 1.0],
        "errors": ["", ""],
    }

    with pytest.raises(AuditError, match="frozen denominator"):
        check_result_accounting(result, 3, "sample")


def test_command_value_rejects_duplicate_flags() -> None:
    with pytest.raises(AuditError, match="exactly one"):
        command_value(["tool", "--seed", "11", "--seed=23"], "--seed")


def test_fallacy_scan_covers_all_eleven_categories() -> None:
    scan = fallacy_scan()

    assert len(scan) == 11
    assert len({item["fallacy"] for item in scan}) == 11
