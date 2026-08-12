"""Build a joint-precision profile from hash-verified structured evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from kvcache.policy import PolicyInputError, validate_joint_precision_profile


class ProfileBuildError(RuntimeError):
    """Raised when a profile recipe cannot be tied to its declared evidence."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileBuildError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProfileBuildError(f"JSON root must be an object: {path}")
    return value


def atomic_write_json_with_hash(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_tmp = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    sidecar_tmp.write_text(f"{digest}\n", encoding="ascii")
    os.replace(sidecar_tmp, sidecar)
    return digest


def _relative_evidence_path(raw_path: Any, repo_root: Path) -> tuple[str, Path]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ProfileBuildError("evidence.path must be a non-empty string")
    declared = Path(raw_path)
    if declared.is_absolute():
        raise ProfileBuildError(f"evidence.path must be repository-relative: {raw_path}")
    resolved_root = repo_root.resolve()
    resolved = (resolved_root / declared).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ProfileBuildError(f"evidence.path escapes the repository: {raw_path}") from error
    return declared.as_posix(), resolved


def _sidecar_digest(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ProfileBuildError(f"evidence SHA-256 sidecar is missing: {sidecar}")
    tokens = sidecar.read_text(encoding="ascii").strip().split()
    if not tokens:
        raise ProfileBuildError(f"evidence SHA-256 sidecar is empty: {sidecar}")
    return tokens[0]


def verify_profile_evidence(profile: Mapping[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    """Recompute every evidence digest and require a matching sidecar."""

    raw_records = profile.get("evidence")
    if not isinstance(raw_records, list) or not raw_records:
        raise ProfileBuildError("profile.evidence must be a non-empty array")
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ProfileBuildError(f"profile.evidence[{index}] must be an object")
        evidence_id = raw_record.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ProfileBuildError(f"profile.evidence[{index}].evidence_id must be a non-empty string")
        if evidence_id in seen:
            raise ProfileBuildError(f"duplicate evidence_id: {evidence_id}")
        seen.add(evidence_id)
        declared_path, resolved_path = _relative_evidence_path(raw_record.get("path"), repo_root)
        if not resolved_path.is_file():
            raise ProfileBuildError(f"evidence file is missing: {resolved_path}")
        observed = sha256_file(resolved_path)
        declared_digest = raw_record.get("sha256")
        if observed != declared_digest:
            raise ProfileBuildError(
                f"evidence digest mismatch for {evidence_id}: declared={declared_digest} observed={observed}"
            )
        sidecar_digest = _sidecar_digest(resolved_path)
        if sidecar_digest != observed:
            raise ProfileBuildError(
                f"evidence sidecar mismatch for {evidence_id}: sidecar={sidecar_digest} observed={observed}"
            )
        verified.append(
            {
                "evidence_id": evidence_id,
                "path": declared_path,
                "sha256": observed,
                "sidecar_path": resolved_path.with_suffix(resolved_path.suffix + ".sha256")
                .relative_to(repo_root.resolve())
                .as_posix(),
            }
        )
    return verified


def resolve_json_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON Pointer without coercing the source value."""

    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ProfileBuildError(f"invalid JSON Pointer: {pointer!r}")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not token.isdigit():
                raise ProfileBuildError(f"array JSON Pointer token is not an index: {token!r}")
            index = int(token)
            if index >= len(current):
                raise ProfileBuildError(f"JSON Pointer index is out of range: {pointer!r}")
            current = current[index]
        elif isinstance(current, dict):
            if token not in current:
                raise ProfileBuildError(f"JSON Pointer key is missing: {pointer!r}")
            current = current[token]
        else:
            raise ProfileBuildError(f"JSON Pointer traverses a scalar: {pointer!r}")
    return copy.deepcopy(current)


def _materialize(value: Any, documents: Mapping[str, Any], used_ids: set[str]) -> Any:
    if isinstance(value, dict) and set(value) == {"source"}:
        source = value["source"]
        if not isinstance(source, dict) or set(source) != {"evidence_id", "json_pointer"}:
            raise ProfileBuildError("source must contain exactly evidence_id and json_pointer")
        evidence_id = source["evidence_id"]
        pointer = source["json_pointer"]
        if not isinstance(evidence_id, str) or evidence_id not in documents:
            raise ProfileBuildError(f"source references unknown evidence: {evidence_id!r}")
        used_ids.add(evidence_id)
        return resolve_json_pointer(documents[evidence_id], pointer)
    if isinstance(value, dict):
        return {key: _materialize(item, documents, used_ids) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item, documents, used_ids) for item in value]
    return copy.deepcopy(value)


def _is_source(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"source"}


def _materialize_rows(
    raw_rows: Any,
    field: str,
    documents: Mapping[str, Any],
    required_sourced_fields: Sequence[str],
) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        raise ProfileBuildError(f"{field} must be an array")
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            raise ProfileBuildError(f"{field}[{index}] must be an object")
        if "evidence_ids" in raw_row:
            raise ProfileBuildError(f"{field}[{index}].evidence_ids is generated and must not be supplied")
        for sourced_field in required_sourced_fields:
            if not _is_source(raw_row.get(sourced_field)):
                raise ProfileBuildError(f"{field}[{index}].{sourced_field} must be sourced from evidence")
        used_ids: set[str] = set()
        row = _materialize(raw_row, documents, used_ids)
        if not used_ids:
            raise ProfileBuildError(f"{field}[{index}] contains no sourced evidence fields")
        row["evidence_ids"] = sorted(used_ids)
        rows.append(row)
    return rows


def build_profile(recipe: Mapping[str, Any], repo_root: Path, *, recipe_path: Path | None = None) -> dict[str, Any]:
    if recipe.get("schema_version") != 1:
        raise ProfileBuildError("recipe.schema_version must be 1")
    raw_evidence = recipe.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise ProfileBuildError("recipe.evidence must be a non-empty array")

    evidence: list[dict[str, Any]] = []
    documents: dict[str, Any] = {}
    for index, raw_record in enumerate(raw_evidence):
        if not isinstance(raw_record, dict):
            raise ProfileBuildError(f"recipe.evidence[{index}] must be an object")
        evidence_id = raw_record.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ProfileBuildError(f"recipe.evidence[{index}].evidence_id must be a non-empty string")
        if evidence_id in documents:
            raise ProfileBuildError(f"duplicate evidence_id: {evidence_id}")
        declared_path, resolved_path = _relative_evidence_path(raw_record.get("path"), repo_root)
        if not resolved_path.is_file():
            raise ProfileBuildError(f"evidence file is missing: {resolved_path}")
        observed_digest = sha256_file(resolved_path)
        expected_digest = raw_record.get("expected_sha256")
        if expected_digest is not None and expected_digest != observed_digest:
            raise ProfileBuildError(
                f"recipe digest mismatch for {evidence_id}: expected={expected_digest} observed={observed_digest}"
            )
        if _sidecar_digest(resolved_path) != observed_digest:
            raise ProfileBuildError(f"evidence sidecar mismatch for {evidence_id}")
        status = raw_record.get("verification_status")
        evidence.append(
            {
                "evidence_id": evidence_id,
                "path": declared_path,
                "sha256": observed_digest,
                "verification_status": status,
            }
        )
        documents[evidence_id] = load_json(resolved_path)

    raw_candidates = recipe.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ProfileBuildError("recipe.candidates must be a non-empty array")
    candidates: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            raise ProfileBuildError(f"recipe.candidates[{index}] must be an object")
        candidate = {
            key: copy.deepcopy(raw_candidate.get(key))
            for key in ("config_id", "kv_cache_dtype", "state_cache_dtype", "deployment")
        }
        candidate["capacity_profiles"] = _materialize_rows(
            raw_candidate.get("capacity_profiles"),
            f"recipe.candidates[{index}].capacity_profiles",
            documents,
            ("cache_bytes", "max_concurrency"),
        )
        candidate["serving_profiles"] = _materialize_rows(
            raw_candidate.get("serving_profiles"),
            f"recipe.candidates[{index}].serving_profiles",
            documents,
            (
                "slo_goodput_lcb_req_s",
                "p95_ttft_ucb_ms",
                "p95_tpot_ucb_ms",
                "n_independent_repeats",
            ),
        )
        candidate["quality_profiles"] = _materialize_rows(
            raw_candidate.get("quality_profiles", []),
            f"recipe.candidates[{index}].quality_profiles",
            documents,
            ("delta_ci95_low", "delta_ci95_high", "n_independent_repeats"),
        )
        candidates.append(candidate)

    profile: dict[str, Any] = {
        "schema_version": 2,
        "profile_status": recipe.get("profile_status"),
        "evidence": evidence,
        "candidates": candidates,
    }
    if "capacity_bytes_semantics" in recipe:
        profile["capacity_bytes_semantics"] = copy.deepcopy(recipe["capacity_bytes_semantics"])
    if "capacity_bytes_source" in recipe:
        profile["capacity_bytes_source"] = copy.deepcopy(recipe["capacity_bytes_source"])
    if recipe_path is not None:
        try:
            relative_recipe = recipe_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError as error:
            raise ProfileBuildError("recipe path must be within the repository") from error
        profile["profile_build"] = {
            "recipe_path": relative_recipe,
            "recipe_sha256": sha256_file(recipe_path),
        }
    try:
        validate_joint_precision_profile(profile)
    except PolicyInputError as error:
        raise ProfileBuildError(f"built profile is invalid: {error}") from error
    verify_profile_evidence(profile, repo_root)
    return profile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    recipe_path = args.recipe.resolve()
    repo_root = args.repo_root.resolve()
    profile = build_profile(load_json(recipe_path), repo_root, recipe_path=recipe_path)
    digest = atomic_write_json_with_hash(args.out.resolve(), profile)
    print(json.dumps({"status": "BUILT", "out": str(args.out.resolve()), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileBuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
