import copy
import json
from pathlib import Path

import pytest

from scripts.eval.validate_ruler_nothink_5cell import ValidationError
from scripts.eval.validate_ruler_nothink_reproduction import (
    compare_reproduction,
    fallacy_scan,
    validate_reproduction_contract,
)


def logical_contract() -> dict:
    return {
        "schema_version": 1,
        "classification": "environment_sensitive_independent_temporal_reproduction",
        "review_policy": {
            "mode": "logical_only",
            "hash_validation_performed": False,
            "script_hashes_are_not_launch_gates": True,
            "result_sidecars_retained": True,
            "failed_or_missing_cells_fail_closed": True,
        },
        "matrix": {
            "cells": [
                ["ruler_fwe", 4096, "2b"],
                ["ruler_fwe", 8192, "2b"],
                ["ruler_niah_multiquery", 4096, "9b"],
                ["ruler_niah_multiquery", 8192, "9b"],
                ["ruler_fwe", 8192, "9b"],
            ],
            "allocations": ["fp16", "fp16_statebf16"],
            "dataset_seeds": [42, 11, 23],
            "engine_seed": 7,
            "expected_cells": 30,
            "expected_samples_per_cell": 20,
        },
        "protocol": {
            "thinking": "disabled",
            "max_tokens": 256,
            "temperature": 0.0,
            "max_model_len": 16384,
            "gpu_memory_utilization": 0.85,
        },
        "execution": {"output_root": "/root/autodl-tmp/results"},
    }


def test_contract_accepts_logical_only_policy(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(logical_contract()), encoding="utf-8")

    assert validate_reproduction_contract(path)["matrix"]["expected_cells"] == 30


def test_contract_rejects_script_hash_gate(tmp_path: Path) -> None:
    contract = logical_contract()
    contract["review_policy"]["hash_validation_performed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValidationError, match="hash validation"):
        validate_reproduction_contract(path)


def result_item(accuracy: float = 100.0) -> dict:
    return {
        "summary": {
            "data_sha256": "a" * 64,
            "ruler_commit": "ruler-revision",
            "vllm_version": "vllm-version",
            "accuracy": accuracy,
        },
        "content_signature": [(1, ("answer",), "answer", (True,), 100, 1)],
    }


def test_compare_reproduction_requires_exact_sample_outputs() -> None:
    key = ("ruler_fwe", 4096, "2b", 42, "fp16")
    original = {key: result_item()}
    reproduction = copy.deepcopy(original)
    reproduction[key]["content_signature"][0] = (1, ("answer",), "other", (False,), 100, 1)

    with pytest.raises(ValidationError, match="sample output mismatch"):
        compare_reproduction(original, reproduction)


def test_fallacy_scan_is_explicit_11_of_11() -> None:
    scan = fallacy_scan()

    assert len(scan) == 11
    assert len({item["fallacy"] for item in scan}) == 11
    assert all(item["status"] == "CHECKED" for item in scan)
