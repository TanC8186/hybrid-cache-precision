import pytest

from scripts.bench.inspect_kv_config import extract_shared_runtime


def test_extract_shared_runtime_requires_consistent_worker_configs() -> None:
    shared = {
        "cache_config": {"mamba_ssm_cache_dtype": "float32"},
        "kv_cache_config": {"num_blocks": 12},
        "capacity": {"tokens": 49152, "max_concurrency": 12.0},
    }
    workers = [{**shared, "rank": rank} for rank in (0, 1)]

    assert extract_shared_runtime(workers) == shared


def test_extract_shared_runtime_rejects_mismatch() -> None:
    shared = {
        "cache_config": {"mamba_ssm_cache_dtype": "float32"},
        "kv_cache_config": {"num_blocks": 12},
        "capacity": {"tokens": 49152, "max_concurrency": 12.0},
    }
    workers = [
        {**shared, "rank": 0},
        {
            **shared,
            "rank": 1,
            "capacity": {"tokens": 40960, "max_concurrency": 10.0},
        },
    ]

    with pytest.raises(RuntimeError, match="configuration mismatch"):
        extract_shared_runtime(workers)
