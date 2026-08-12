"""Build a selector profile whose memory field uses physical backing bytes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.controller.build_joint_precision_calibration_recipe import (
    EXPECTED_ALLOCATIONS,
    derive_physical_cache_summary,
    load_verified_json,
    repo_relative,
)
from scripts.controller.build_joint_precision_profile import (
    ProfileBuildError,
    atomic_write_json_with_hash,
    build_profile,
    load_json,
)
from kvcache.policy import CAPACITY_BYTES_SEMANTICS


SEMANTICS = CAPACITY_BYTES_SEMANTICS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileBuildError(message)


def _resolve_repo_path(raw_path: Any, repo_root: Path, field: str) -> Path:
    _require(isinstance(raw_path, str) and raw_path.strip(), f"{field} must be a path")
    declared = Path(raw_path)
    _require(not declared.is_absolute(), f"{field} must be repository-relative")
    resolved_root = repo_root.resolve()
    resolved = (resolved_root / declared).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ProfileBuildError(f"{field} escapes the repository") from error
    return resolved


def build_physical_evidence(recipe: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    records = recipe.get("evidence")
    _require(isinstance(records, list), "recipe.evidence must be an array")
    by_id = {record.get("evidence_id"): record for record in records if isinstance(record, dict)}
    allocations: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, str]] = {}
    for allocation in EXPECTED_ALLOCATIONS:
        evidence_id = f"capacity_{allocation}"
        record = by_id.get(evidence_id)
        _require(isinstance(record, dict), f"missing {evidence_id} evidence")
        path = _resolve_repo_path(record.get("path"), repo_root, f"evidence.{evidence_id}.path")
        document, observed_sha = load_verified_json(path)
        expected_sha = record.get("expected_sha256")
        _require(expected_sha in (None, observed_sha), f"{evidence_id} digest drift")
        allocations[allocation] = derive_physical_cache_summary(allocation, document)
        sources[allocation] = {
            "path": repo_relative(path, repo_root),
            "sha256": observed_sha,
        }

    return {
        "schema_version": 1,
        "material_passport": {
            "origin_skill": "experiment-skill",
            "origin_mode": "validate",
            "origin_date": datetime.now(timezone.utc).date().isoformat(),
            "verification_status": "VERIFIED",
            "version_label": "physical_cache_evidence_v1",
        },
        "capacity_bytes_semantics": SEMANTICS,
        "source_documents": sources,
        "allocations": allocations,
    }


def build_physical_recipe(
    base_recipe: Mapping[str, Any],
    *,
    physical_evidence_path: Path,
    physical_evidence_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    recipe = copy.deepcopy(dict(base_recipe))
    evidence = recipe.get("evidence")
    _require(isinstance(evidence, list), "recipe.evidence must be an array")
    _require(
        all(record.get("evidence_id") != "capacity_physical" for record in evidence if isinstance(record, dict)),
        "recipe already contains capacity_physical evidence",
    )
    evidence.append(
        {
            "evidence_id": "capacity_physical",
            "path": repo_relative(physical_evidence_path, repo_root),
            "expected_sha256": physical_evidence_sha256,
            "verification_status": "VERIFIED",
        }
    )
    recipe["capacity_bytes_semantics"] = SEMANTICS
    recipe["capacity_bytes_source"] = "kv_cache_config.tensors[].size summed by unique tensor_id"

    candidates = recipe.get("candidates")
    _require(isinstance(candidates, list), "recipe.candidates must be an array")
    for candidate in candidates:
        _require(isinstance(candidate, dict), "recipe candidate must be an object")
        allocation = candidate.get("config_id")
        _require(allocation in EXPECTED_ALLOCATIONS, f"unknown candidate allocation: {allocation}")
        rows = candidate.get("capacity_profiles")
        _require(isinstance(rows, list) and len(rows) == 1, f"{allocation} must have one capacity row")
        rows[0]["cache_bytes"] = {
            "source": {
                "evidence_id": "capacity_physical",
                "json_pointer": f"/allocations/{allocation}/physical_cache_bytes",
            }
        }
    return recipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-recipe", type=Path, required=True)
    parser.add_argument("--physical-evidence-out", type=Path, required=True)
    parser.add_argument("--recipe-out", type=Path, required=True)
    parser.add_argument("--profile-out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    base_recipe_path = args.base_recipe.resolve()
    evidence_path = args.physical_evidence_out.resolve()
    recipe_path = args.recipe_out.resolve()
    profile_path = args.profile_out.resolve()
    for path in (evidence_path, recipe_path, profile_path, evidence_path.with_suffix(evidence_path.suffix + ".sha256")):
        _require(not path.exists(), f"refusing to overwrite output: {path}")

    base_recipe = load_json(base_recipe_path)
    physical_evidence = build_physical_evidence(base_recipe, repo_root)
    physical_sha = atomic_write_json_with_hash(evidence_path, physical_evidence)
    recipe = build_physical_recipe(
        base_recipe,
        physical_evidence_path=evidence_path,
        physical_evidence_sha256=physical_sha,
        repo_root=repo_root,
    )
    atomic_write_json_with_hash(recipe_path, recipe)
    profile = build_profile(recipe, repo_root, recipe_path=recipe_path)
    profile_sha = atomic_write_json_with_hash(profile_path, profile)
    print(
        json.dumps(
            {
                "status": "BUILT",
                "physical_evidence": str(evidence_path),
                "physical_evidence_sha256": physical_sha,
                "recipe": str(recipe_path),
                "profile": str(profile_path),
                "profile_sha256": profile_sha,
                "capacity_bytes_semantics": SEMANTICS,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileBuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
