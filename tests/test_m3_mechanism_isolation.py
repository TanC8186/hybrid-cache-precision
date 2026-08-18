from __future__ import annotations

from pathlib import Path

from scripts.analyze.analyze_m3_mechanism_isolation import fallacy_scan, metric_families
from scripts.bench.run_steady_state import ServerSession, build_sample_plan, load_config, resolve_phase


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "m3_mechanism_isolation_2b.yaml"


def test_m3_phase_sizes_and_disjoint_seeds() -> None:
    config = load_config(CONFIG)
    assert len(build_sample_plan(config, resolve_phase(config, "mvex"))) == 2
    assert len(build_sample_plan(config, resolve_phase(config, "pilot"))) == 6
    formal = build_sample_plan(config, resolve_phase(config, "formal"))
    assert len(formal) == 18
    assert {sample["seed"] for sample in formal} == {11, 23, 47}
    assert {sample["request_rate"] for sample in formal} == {40.0}


def test_m3_pairs_are_orthogonal_by_constraint() -> None:
    config = load_config(CONFIG)
    allocations = config["allocations"]
    for constraint in ("block_count", "bytes", "concurrency"):
        full = allocations[f"full_fixed_{constraint}"]
        joint = allocations[f"joint_fixed_{constraint}"]
        assert full["constraint"] == f"fixed_{constraint}"
        assert joint["constraint"] == f"fixed_{constraint}"
        assert full["server_args"] != joint["server_args"]
    assert config["protocol"]["telemetry_required_metrics"]


def test_m3_fallacy_scan_covers_all_eleven_categories() -> None:
    scan = fallacy_scan()

    assert len(scan) == 11
    assert len({item["fallacy"] for item in scan}) == 11
    assert all(item["status"] == "CHECKED" for item in scan)


def test_prometheus_histogram_samples_resolve_to_required_families() -> None:
    payload = """vllm:time_to_first_token_seconds_bucket{le=\"0.5\"} 3
vllm:time_to_first_token_seconds_count 3
vllm:time_to_first_token_seconds_sum 0.4
vllm:kv_cache_usage_perc 0.25"""
    parsed = ServerSession._parse_prometheus_metrics(payload)
    assert "vllm:time_to_first_token_seconds" in parsed
    assert "vllm:kv_cache_usage_perc" in parsed
    assert "vllm:time_to_first_token_seconds" in metric_families(set(parsed))
