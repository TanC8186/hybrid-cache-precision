from scripts.analyze.steady_state import summarize


def test_summarize_three_seed_ci_uses_small_sample_t() -> None:
    result = summarize([30.0, 32.0, 34.0])

    assert result["n"] == 3
    assert result["mean"] == 32.0
    assert result["std"] == 2.0
    assert 4.9 < result["ci95_half_width"] < 5.0


def test_single_seed_summary_does_not_invent_ci() -> None:
    result = summarize([30.0])

    assert result == {
        "n": 1,
        "mean": 30.0,
        "std": 0.0,
        "ci95_half_width": None,
    }
