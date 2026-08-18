from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.controller.build_joint_precision_calibration_recipe import (
    EXPECTED_ALLOCATIONS,
    ProfileBuildError,
    build_recipe,
    derive_physical_cache_summary,
    normalize_quality_evidence,
    parse_capacity_paths,
)
from scripts.controller.build_joint_precision_profile import build_profile


def write_evidence(repo_root: Path, relative_path: str, value: dict) -> Path:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}\n", encoding="ascii")
    return path


def calibration_document() -> dict:
    profile_inputs = {}
    for index, allocation in enumerate(EXPECTED_ALLOCATIONS):
        profile_inputs[allocation] = {
            "random": {
                "40": {
                    "offered_rate_req_s": 40.0,
                    "n_independent_repeats": 3,
                    "p95_ttft_ucb_ms": 400.0 + index,
                    "p95_tpot_ucb_ms": 100.0 + index,
                    "slo_sweep": {
                        "250": {
                            "slo_goodput_lcb_req_s": -1.0,
                            "profile_eligible": False,
                        },
                        "500": {
                            "slo_goodput_lcb_req_s": 30.0 + index,
                            "profile_eligible": True,
                        },
                    },
                }
            }
        }
    return {
        "gate": "PASS",
        "evidence_status": "ANALYZED",
        "attempt_id": "calibration-attempt",
        "fallacy_scan": {"coverage": "11/11"},
        "aggregation": {"profile_inputs": profile_inputs},
    }


def capacity_document(allocation: str, index: int) -> dict:
    metadata = EXPECTED_ALLOCATIONS[allocation]
    return {
        "args": {
            "kv_cache_dtype": metadata["kv_cache_dtype"],
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.8,
        },
        "resolved_mamba_ssm_cache_dtype": metadata["state_cache_dtype"],
        "capacity": {"max_concurrency": 200.0 + index},
        "cache_tensor_summary": {"workers": [{"total_cache_bytes": 1000 + index}]},
    }


def normalized_quality_document() -> dict:
    return {
        "material_passport": {"verification_status": "ANALYZED"},
        "task": "gsm8k",
        "candidates": {
            allocation: {
                "delta_ci95_low": -0.01 * index,
                "delta_ci95_high": 0.01 * index,
                "inference_method": "intercept_only_ols_two_way_cluster_robust_cr1",
                "estimand": "mean paired accuracy difference over observed seed-item draws",
                "n_seed_item_draws": 1800,
                "n_item_clusters": 1017,
                "n_seed_clusters": 9,
                "cluster_degrees_of_freedom": 8,
            }
            for index, allocation in enumerate(EXPECTED_ALLOCATIONS)
        },
    }


def test_normalize_quality_evidence_maps_all_four_allocations() -> None:
    rows = []
    for index, metadata in enumerate(EXPECTED_ALLOCATIONS.values()):
        row = {
            "allocation": metadata["quality_allocation"],
            "n_dataset_seeds": 9,
            "cluster_robust_inference": {
                "method": "intercept_only_ols_two_way_cluster_robust_cr1",
                "n_item_clusters": 1017,
                "n_seed_clusters": 9,
                "degrees_of_freedom": 8,
            },
        }
        if index:
            row["ci95_vs_fp16"] = [-0.01 * index, 0.01 * index]
        rows.append(row)

    normalized = normalize_quality_evidence(
        {
            "schema_version": 2,
            "bench": "gsm8k",
            "attempt": "quality",
            "protocol": "paired",
            "primary_estimand": "mean paired accuracy difference over observed seed-item draws",
            "diagnostics": {"n_seed_item_draws": 1800, "n_unique_items": 1017},
            "rows": rows,
        },
        source_path="results/quality.json",
        source_sha256="a" * 64,
    )

    assert set(normalized["candidates"]) == set(EXPECTED_ALLOCATIONS)
    assert normalized["candidates"]["full"]["delta_ci95_low"] == 0.0
    assert normalized["candidates"]["joint"]["delta_ci95_high"] == pytest.approx(0.03)
    assert normalized["source"]["sha256"] == "a" * 64
    assert normalized["candidates"]["full"]["n_seed_item_draws"] == 1800
    assert normalized["candidates"]["joint"]["cluster_degrees_of_freedom"] == 8


def test_recipe_materializes_a_valid_calibration_profile(tmp_path: Path) -> None:
    calibration = write_evidence(tmp_path, "evidence/calibration.json", calibration_document())
    frozen = write_evidence(
        tmp_path,
        "evidence/frozen.json",
        {
            "contract_status": "FROZEN",
            "attempt_id": "calibration-attempt",
            "model": {"id": "Qwen/Qwen3.5-2B"},
            "matrix": {"tpot_threshold_ms": 200},
        },
    )
    validation = write_evidence(
        tmp_path,
        "evidence/capacity-validation.json",
        {
            "material_passport": {"verification_status": "VERIFIED"},
            "reproducibility_verdict": "REPRODUCIBLE",
        },
    )
    quality = write_evidence(tmp_path, "evidence/quality.json", normalized_quality_document())
    capacity_paths = {
        allocation: write_evidence(
            tmp_path,
            f"evidence/capacity-{allocation}.json",
            capacity_document(allocation, index),
        )
        for index, allocation in enumerate(EXPECTED_ALLOCATIONS)
    }

    recipe = build_recipe(
        calibration_path=calibration,
        frozen_contract_path=frozen,
        capacity_validation_path=validation,
        capacity_paths=capacity_paths,
        quality_evidence_path=quality,
        repo_root=tmp_path,
    )
    profile = build_profile(recipe, tmp_path)

    assert profile["profile_status"] == "CALIBRATION"
    assert len(profile["candidates"]) == 4
    assert sum(len(row["serving_profiles"]) for row in profile["candidates"]) == 4
    assert profile["candidates"][0]["serving_profiles"][0]["slo"]["p95_ttft_ms"] == 500.0
    assert profile["candidates"][0]["capacity_profiles"][0]["cache_bytes"] == 1000
    assert profile["candidates"][3]["quality_profiles"][0]["delta_ci95_low"] == pytest.approx(-0.03)


def test_parse_capacity_paths_requires_exact_allocation_set() -> None:
    with pytest.raises(ProfileBuildError, match="exactly four"):
        parse_capacity_paths(["full=full.json"])

    with pytest.raises(ProfileBuildError, match="duplicate"):
        parse_capacity_paths(
            [
                "full=full-a.json",
                "full=full-b.json",
                "kv_only=kv.json",
                "state_only=state.json",
                "joint=joint.json",
            ]
        )


def test_derive_physical_cache_summary_uses_unique_backing_storage() -> None:
    document = capacity_document("full", 0)
    document["kv_cache_config"] = {
        "num_blocks": 2,
        "tensors": [
            {"tensor_id": 0, "size": 600},
            {"tensor_id": 1, "size": 300},
        ],
    }
    summary = derive_physical_cache_summary("full", document)

    assert summary["physical_cache_bytes"] == 900
    assert summary["logical_view_bytes"] == 1000
    assert summary["backing_storage_count"] == 2


def test_derive_physical_cache_summary_rejects_duplicate_backing_storage_ids() -> None:
    document = capacity_document("full", 0)
    document["kv_cache_config"] = {
        "num_blocks": 2,
        "tensors": [
            {"tensor_id": 0, "size": 600},
            {"tensor_id": 0, "size": 300},
        ],
    }

    with pytest.raises(ProfileBuildError, match="duplicate backing-storage"):
        derive_physical_cache_summary("full", document)
