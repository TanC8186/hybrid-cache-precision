from scripts.analyze.verify_a2_reproduction import (
    classify_verdict,
    group_types,
    symmetric_relative_difference,
)


def test_symmetric_relative_difference_uses_larger_magnitude() -> None:
    assert symmetric_relative_difference(10.0, 9.0) == 0.1
    assert symmetric_relative_difference(0.0, 0.0) == 0.0


def test_environment_sensitive_mismatch_is_partial_under_frozen_exact_gate() -> None:
    assert (
        classify_verdict(
            structural_pass=True,
            exact_capacity_match=False,
            within_environment_tolerance=True,
        )
        == "PARTIALLY_REPRODUCIBLE"
    )


def test_structural_failure_is_not_reproducible() -> None:
    assert (
        classify_verdict(
            structural_pass=False,
            exact_capacity_match=True,
            within_environment_tolerance=True,
        )
        == "NOT_REPRODUCIBLE"
    )


def test_group_types_reads_serialized_spec_type() -> None:
    report = {
        "kv_cache_config": {
            "groups": [
                {"spec": {"type": "UniformTypeKVCacheSpecs"}},
                {"spec": {"type": "MambaSpec"}},
            ]
        }
    }

    assert group_types(report) == ["UniformTypeKVCacheSpecs", "MambaSpec"]
