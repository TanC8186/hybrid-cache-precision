from scripts.analyze.verify_e3_reproducibility import (
    VerificationError,
    derive_boundaries,
    merge_points,
    symmetric_relative_difference,
)


def test_symmetric_relative_difference_uses_larger_magnitude() -> None:
    assert symmetric_relative_difference(10.0, 9.0) == 0.1
    assert symmetric_relative_difference(0.0, 0.0) == 0.0


def test_boundary_derivation_retains_right_censoring() -> None:
    key = ("int4", "random", 7, 1000.0)
    boundaries = derive_boundaries({key: [(35.0, True), (40.0, True)]})

    assert boundaries[key] == {
        "max_tested_sustainable_rate": 40.0,
        "right_censored": True,
        "tested_rates": [35.0, 40.0],
    }


def test_upper_neighbor_removes_right_censoring_without_pooling_samples() -> None:
    key = ("fp16", "sharegpt", 7, 1000.0)
    lower = {key: [(20.0, True), (25.0, True), (30.0, True), (35.0, True)]}
    upper = {key: [(40.0, False)]}

    combined = derive_boundaries(merge_points(lower, upper))

    assert combined[key]["max_tested_sustainable_rate"] == 35.0
    assert combined[key]["right_censored"] is False


def test_conflicting_duplicate_boundary_point_fails_closed() -> None:
    key = ("fp16", "sharegpt", 7, 1000.0)

    try:
        merge_points({key: [(40.0, True)]}, {key: [(40.0, False)]})
    except VerificationError as exc:
        assert "conflicting boundary point" in str(exc)
    else:
        raise AssertionError("expected conflicting duplicate point to fail")
