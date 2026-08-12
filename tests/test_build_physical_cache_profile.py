from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.controller.build_joint_precision_calibration_recipe import EXPECTED_ALLOCATIONS
from scripts.controller.build_physical_cache_profile import (
    SEMANTICS,
    build_physical_evidence,
    build_physical_recipe,
)


def write_json_with_hash(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest


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
        "kv_cache_config": {
            "num_blocks": 2,
            "tensors": [
                {"tensor_id": 0, "size": 600 + index},
                {"tensor_id": 1, "size": 300},
            ],
        },
    }


def base_recipe(tmp_path: Path) -> tuple[dict, dict[str, str]]:
    evidence = []
    hashes = {}
    for index, allocation in enumerate(EXPECTED_ALLOCATIONS):
        relative = f"evidence/{allocation}.json"
        digest = write_json_with_hash(tmp_path / relative, capacity_document(allocation, index))
        hashes[allocation] = digest
        evidence.append(
            {
                "evidence_id": f"capacity_{allocation}",
                "path": relative,
                "expected_sha256": digest,
                "verification_status": "VERIFIED",
            }
        )
    recipe = {
        "schema_version": 1,
        "profile_status": "CALIBRATION",
        "evidence": evidence,
        "candidates": [
            {
                "config_id": allocation,
                "capacity_profiles": [{"cache_bytes": {}, "max_concurrency": {}}],
            }
            for allocation in EXPECTED_ALLOCATIONS
        ],
    }
    return recipe, hashes


def test_physical_evidence_records_unique_storage_and_source_hashes(tmp_path: Path) -> None:
    recipe, hashes = base_recipe(tmp_path)
    evidence = build_physical_evidence(recipe, tmp_path)

    assert evidence["capacity_bytes_semantics"] == SEMANTICS
    assert evidence["allocations"]["full"]["physical_cache_bytes"] == 900
    assert evidence["source_documents"]["full"]["sha256"] == hashes["full"]


def test_physical_recipe_repoints_only_capacity_bytes(tmp_path: Path) -> None:
    recipe, _ = base_recipe(tmp_path)
    physical_path = tmp_path / "evidence" / "physical.json"
    physical_sha = write_json_with_hash(physical_path, {"allocations": {}})

    updated = build_physical_recipe(
        recipe,
        physical_evidence_path=physical_path,
        physical_evidence_sha256=physical_sha,
        repo_root=tmp_path,
    )

    assert updated["capacity_bytes_semantics"] == SEMANTICS
    assert updated["evidence"][-1]["evidence_id"] == "capacity_physical"
    for candidate in updated["candidates"]:
        source = candidate["capacity_profiles"][0]["cache_bytes"]["source"]
        assert source["evidence_id"] == "capacity_physical"
        assert source["json_pointer"].startswith(f"/allocations/{candidate['config_id']}/")
