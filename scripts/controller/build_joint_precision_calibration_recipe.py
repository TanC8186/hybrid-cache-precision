"""Build a profile recipe from audited calibration, capacity, and quality evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from kvcache.policy import canonical_precision_args
from scripts.controller.build_joint_precision_profile import (
    ProfileBuildError,
    atomic_write_json_with_hash,
    load_json,
    sha256_file,
)

EXPECTED_ALLOCATIONS = {
    "full": {
        "kv_cache_dtype": "auto",
        "state_cache_dtype": "float32",
        "quality_allocation": "fp16",
    },
    "kv_only": {
        "kv_cache_dtype": "int4_per_token_head",
        "state_cache_dtype": "float32",
        "quality_allocation": "uniform_int4",
    },
    "state_only": {
        "kv_cache_dtype": "auto",
        "state_cache_dtype": "bfloat16",
        "quality_allocation": "fp16_statebf16",
    },
    "joint": {
        "kv_cache_dtype": "int4_per_token_head",
        "state_cache_dtype": "bfloat16",
        "quality_allocation": "uniform_int4_statebf16",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileBuildError(message)


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise ProfileBuildError(f"evidence path must be inside the repository: {path}") from error


def load_verified_json(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    require(resolved.is_file(), f"evidence file is missing: {resolved}")
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    require(sidecar.is_file(), f"evidence sidecar is missing: {sidecar}")
    tokens = sidecar.read_text(encoding="ascii").strip().split()
    require(bool(tokens), f"evidence sidecar is empty: {sidecar}")
    digest = sha256_file(resolved)
    require(tokens[0] == digest, f"evidence sidecar mismatch: {resolved}")
    return load_json(resolved), digest


def json_pointer(*tokens: str) -> str:
    escaped = [str(token).replace("~", "~0").replace("/", "~1") for token in tokens]
    return "/" + "/".join(escaped)


def source(evidence_id: str, *tokens: str) -> dict[str, Any]:
    return {
        "source": {
            "evidence_id": evidence_id,
            "json_pointer": json_pointer(*tokens),
        }
    }


def parse_capacity_paths(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        allocation, separator, raw_path = value.partition("=")
        require(bool(separator) and bool(raw_path), "--capacity-evidence must use allocation=path")
        require(allocation in EXPECTED_ALLOCATIONS, f"unknown capacity allocation: {allocation}")
        require(allocation not in parsed, f"duplicate capacity allocation: {allocation}")
        parsed[allocation] = Path(raw_path)
    require(set(parsed) == set(EXPECTED_ALLOCATIONS), "capacity evidence must cover exactly four allocations")
    return parsed


def normalize_quality_evidence(
    quality: Mapping[str, Any],
    *,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    require(quality.get("bench") == "gsm8k", "quality evidence must be the GSM8K analysis")
    require(
        quality.get("schema_version") == 2,
        "quality evidence must use the dependence-aware schema_version=2 analysis",
    )
    diagnostics = quality.get("diagnostics")
    require(isinstance(diagnostics, dict), "quality diagnostics must be an object")
    n_draws = diagnostics.get("n_seed_item_draws")
    n_items = diagnostics.get("n_unique_items")
    require(isinstance(n_draws, int) and n_draws > 0, "invalid seed-item draw count")
    require(isinstance(n_items, int) and n_items > 1, "invalid item-cluster count")
    raw_rows = quality.get("rows")
    require(isinstance(raw_rows, list), "quality rows must be an array")
    indexed: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, raw_row in enumerate(raw_rows):
        require(isinstance(raw_row, dict), f"quality row {index} must be an object")
        allocation = raw_row.get("allocation")
        require(isinstance(allocation, str), f"quality row {index} has no allocation")
        require(allocation not in indexed, f"duplicate quality allocation: {allocation}")
        indexed[allocation] = (index, raw_row)

    candidates: dict[str, Any] = {}
    for allocation, metadata in EXPECTED_ALLOCATIONS.items():
        quality_allocation = str(metadata["quality_allocation"])
        require(quality_allocation in indexed, f"quality allocation is missing: {quality_allocation}")
        source_index, row = indexed[quality_allocation]
        n = row.get("n_dataset_seeds")
        require(
            isinstance(n, int) and not isinstance(n, bool) and n >= 2,
            f"invalid dataset-seed count: {quality_allocation}",
        )
        if allocation == "full":
            interval = [0.0, 0.0]
        else:
            interval = row.get("ci95_vs_fp16")
            require(
                isinstance(interval, list) and len(interval) == 2,
                f"quality CI is missing: {quality_allocation}",
            )
        low, high = (float(interval[0]), float(interval[1]))
        require(low <= high, f"quality CI is reversed: {quality_allocation}")
        if allocation == "full":
            inference = next(
                (
                    candidate.get("cluster_robust_inference")
                    for _, candidate in indexed.values()
                    if candidate.get("cluster_robust_inference") is not None
                ),
                None,
            )
        else:
            inference = row.get("cluster_robust_inference")
        require(isinstance(inference, dict), f"cluster inference is missing: {quality_allocation}")
        method = inference.get("method")
        n_seed_clusters = inference.get("n_seed_clusters")
        n_item_clusters = inference.get("n_item_clusters")
        cluster_df = inference.get("degrees_of_freedom")
        require(isinstance(method, str) and method, f"invalid inference method: {quality_allocation}")
        require(n_seed_clusters == n, f"seed-cluster count mismatch: {quality_allocation}")
        require(n_item_clusters == n_items, f"item-cluster count mismatch: {quality_allocation}")
        require(isinstance(cluster_df, int) and cluster_df > 0, f"invalid cluster df: {quality_allocation}")
        candidates[allocation] = {
            "source_allocation": quality_allocation,
            "source_row_index": source_index,
            "delta_ci95_low": low,
            "delta_ci95_high": high,
            "inference_method": method,
            "estimand": quality.get("primary_estimand"),
            "n_seed_item_draws": n_draws,
            "n_item_clusters": n_item_clusters,
            "n_seed_clusters": n_seed_clusters,
            "cluster_degrees_of_freedom": cluster_df,
        }

    return {
        "schema_version": 2,
        "material_passport": {
            "origin_skill": "experiment-skill",
            "origin_mode": "validate",
            "origin_date": datetime.now(timezone.utc).date().isoformat(),
            "verification_status": "ANALYZED",
            "version_label": "joint_precision_quality_evidence_v2",
        },
        "source": {
            "path": source_path,
            "sha256": source_sha256,
            "attempt": quality.get("attempt"),
            "protocol": quality.get("protocol"),
        },
        "task": "gsm8k",
        "candidates": candidates,
    }


def audit_capacity_document(
    allocation: str,
    document: Mapping[str, Any],
) -> tuple[int, float]:
    metadata = EXPECTED_ALLOCATIONS[allocation]
    args = document.get("args")
    capacity = document.get("capacity")
    summaries = document.get("cache_tensor_summary")
    require(isinstance(args, dict), f"{allocation}: capacity args are missing")
    require(isinstance(capacity, dict), f"{allocation}: capacity result is missing")
    require(isinstance(summaries, dict), f"{allocation}: cache summary is missing")
    workers = summaries.get("workers")
    require(isinstance(workers, list) and len(workers) == 1, f"{allocation}: expected one capacity worker")
    worker = workers[0]
    require(isinstance(worker, dict), f"{allocation}: worker summary is invalid")
    require(
        args.get("kv_cache_dtype") == metadata["kv_cache_dtype"],
        f"{allocation}: KV precision mismatch",
    )
    require(
        document.get("resolved_mamba_ssm_cache_dtype") == metadata["state_cache_dtype"],
        f"{allocation}: state precision mismatch",
    )
    max_model_len = args.get("max_model_len")
    utilization = args.get("gpu_memory_utilization")
    require(isinstance(max_model_len, int) and max_model_len > 0, f"{allocation}: invalid max model length")
    require(isinstance(utilization, (int, float)) and 0 < float(utilization) <= 1, f"{allocation}: invalid utilization")
    require(
        isinstance(worker.get("total_cache_bytes"), int) and worker["total_cache_bytes"] > 0,
        f"{allocation}: invalid cache bytes",
    )
    require(
        isinstance(capacity.get("max_concurrency"), (int, float)) and float(capacity["max_concurrency"]) > 0,
        f"{allocation}: invalid allocator-equivalent sequence slots",
    )
    return max_model_len, float(utilization)


def derive_physical_cache_summary(allocation: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """Derive physical cache bytes from unique backing-storage tensors.

    ``cache_tensor_summary`` reports per-layer logical views.  Hybrid layers
    share those views, so summing that field double-counts the allocation.  The
    ``kv_cache_config.tensors`` entries are the unique backing storages used by
    the allocator and are therefore the only valid source for a memory budget.
    """

    audit_capacity_document(allocation, document)
    cache_config = document.get("kv_cache_config")
    require(isinstance(cache_config, dict), f"{allocation}: kv cache config is missing")
    raw_tensors = cache_config.get("tensors")
    require(isinstance(raw_tensors, list) and raw_tensors, f"{allocation}: backing-storage tensors are missing")

    tensor_ids: list[int] = []
    sizes: list[int] = []
    for index, raw_tensor in enumerate(raw_tensors):
        require(isinstance(raw_tensor, dict), f"{allocation}: tensor {index} is invalid")
        tensor_id = raw_tensor.get("tensor_id")
        size = raw_tensor.get("size")
        require(
            isinstance(tensor_id, int) and not isinstance(tensor_id, bool) and tensor_id >= 0,
            f"{allocation}: tensor {index} has an invalid tensor_id",
        )
        require(
            isinstance(size, int) and not isinstance(size, bool) and size > 0,
            f"{allocation}: tensor {index} has an invalid backing-storage size",
        )
        tensor_ids.append(tensor_id)
        sizes.append(size)

    require(len(tensor_ids) == len(set(tensor_ids)), f"{allocation}: duplicate backing-storage tensor_id")
    logical_bytes = document["cache_tensor_summary"]["workers"][0]["total_cache_bytes"]
    physical_bytes = sum(sizes)
    require(logical_bytes >= physical_bytes, f"{allocation}: logical cache bytes are below physical bytes")
    return {
        "physical_cache_bytes": physical_bytes,
        "logical_view_bytes": logical_bytes,
        "backing_storage_count": len(sizes),
        "backing_storage_tensor_ids": tensor_ids,
        "allocator_equivalent_sequence_slots": document["capacity"]["max_concurrency"],
        "max_model_len": document["args"]["max_model_len"],
        "gpu_memory_utilization": document["args"]["gpu_memory_utilization"],
        "num_gpu_blocks": cache_config.get("num_blocks"),
    }


def serving_rows(allocation: str, profile_inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    allocation_inputs = profile_inputs.get(allocation)
    require(isinstance(allocation_inputs, dict), f"calibration profile inputs are missing: {allocation}")
    rows: list[dict[str, Any]] = []
    for workload in sorted(allocation_inputs):
        rates = allocation_inputs[workload]
        require(isinstance(rates, dict), f"invalid workload inputs: {allocation}/{workload}")
        for rate_key in sorted(rates, key=float):
            point = rates[rate_key]
            require(isinstance(point, dict), f"invalid rate input: {allocation}/{workload}/{rate_key}")
            sweep = point.get("slo_sweep")
            require(isinstance(sweep, dict), f"SLO sweep is missing: {allocation}/{workload}/{rate_key}")
            for threshold in sorted(sweep, key=float):
                threshold_row = sweep[threshold]
                require(isinstance(threshold_row, dict), "invalid SLO sweep row")
                if threshold_row.get("profile_eligible") is not True:
                    continue
                rows.append(
                    {
                        "model_id": None,
                        "max_model_len": None,
                        "workload": workload,
                        "offered_rate_req_s": float(rate_key),
                        "slo": {"p95_ttft_ms": float(threshold), "p95_tpot_ms": 200.0},
                        "slo_goodput_lcb_req_s": source(
                            "serving_calibration",
                            "aggregation",
                            "profile_inputs",
                            allocation,
                            workload,
                            rate_key,
                            "slo_sweep",
                            threshold,
                            "slo_goodput_lcb_req_s",
                        ),
                        "p95_ttft_ucb_ms": source(
                            "serving_calibration",
                            "aggregation",
                            "profile_inputs",
                            allocation,
                            workload,
                            rate_key,
                            "p95_ttft_ucb_ms",
                        ),
                        "p95_tpot_ucb_ms": source(
                            "serving_calibration",
                            "aggregation",
                            "profile_inputs",
                            allocation,
                            workload,
                            rate_key,
                            "p95_tpot_ucb_ms",
                        ),
                        "n_independent_repeats": source(
                            "serving_calibration",
                            "aggregation",
                            "profile_inputs",
                            allocation,
                            workload,
                            rate_key,
                            "n_independent_repeats",
                        ),
                    }
                )
    require(bool(rows), f"no profile-eligible serving rows: {allocation}")
    return rows


def build_recipe(
    *,
    calibration_path: Path,
    frozen_contract_path: Path,
    capacity_validation_path: Path,
    capacity_paths: Mapping[str, Path],
    quality_evidence_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    calibration, calibration_sha = load_verified_json(calibration_path)
    frozen, frozen_sha = load_verified_json(frozen_contract_path)
    capacity_validation, capacity_validation_sha = load_verified_json(capacity_validation_path)
    quality, quality_sha = load_verified_json(quality_evidence_path)
    require(calibration.get("gate") == "PASS", "calibration analysis did not pass")
    require(calibration.get("evidence_status") == "ANALYZED", "calibration evidence status is not ANALYZED")
    require(calibration.get("fallacy_scan", {}).get("coverage") == "11/11", "calibration fallacy scan is incomplete")
    require(frozen.get("contract_status") == "FROZEN", "calibration contract is not frozen")
    require(calibration.get("attempt_id") == frozen.get("attempt_id"), "calibration attempt ID drift")
    require(
        capacity_validation.get("material_passport", {}).get("verification_status") == "VERIFIED"
        and capacity_validation.get("reproducibility_verdict") == "REPRODUCIBLE",
        "capacity validation is not reproducible VERIFIED evidence",
    )
    require(
        quality.get("material_passport", {}).get("verification_status") == "ANALYZED",
        "quality evidence is not ANALYZED",
    )
    require(quality.get("task") == "gsm8k", "normalized quality task is not GSM8K")

    capacity_documents: dict[str, dict[str, Any]] = {}
    capacity_hashes: dict[str, str] = {}
    strata: set[tuple[int, float]] = set()
    for allocation in EXPECTED_ALLOCATIONS:
        document, digest = load_verified_json(capacity_paths[allocation])
        capacity_documents[allocation] = document
        capacity_hashes[allocation] = digest
        strata.add(audit_capacity_document(allocation, document))
    require(len(strata) == 1, "capacity evidence does not share one model-length/utilization stratum")
    max_model_len, utilization = next(iter(strata))
    model_id = frozen.get("model", {}).get("id")
    require(isinstance(model_id, str) and bool(model_id), "frozen model ID is missing")
    profile_inputs = calibration.get("aggregation", {}).get("profile_inputs")
    require(isinstance(profile_inputs, dict), "calibration profile inputs are missing")

    evidence = [
        {
            "evidence_id": "serving_calibration",
            "path": repo_relative(calibration_path, repo_root),
            "expected_sha256": calibration_sha,
            "verification_status": "ANALYZED",
        },
        {
            "evidence_id": "calibration_contract",
            "path": repo_relative(frozen_contract_path, repo_root),
            "expected_sha256": frozen_sha,
            "verification_status": "ANALYZED",
        },
        {
            "evidence_id": "capacity_validation",
            "path": repo_relative(capacity_validation_path, repo_root),
            "expected_sha256": capacity_validation_sha,
            "verification_status": "VERIFIED",
        },
        {
            "evidence_id": "quality_gsm8k",
            "path": repo_relative(quality_evidence_path, repo_root),
            "expected_sha256": quality_sha,
            "verification_status": "ANALYZED",
        },
    ]
    for allocation in EXPECTED_ALLOCATIONS:
        evidence.append(
            {
                "evidence_id": f"capacity_{allocation}",
                "path": repo_relative(capacity_paths[allocation], repo_root),
                "expected_sha256": capacity_hashes[allocation],
                "verification_status": "VERIFIED",
            }
        )

    candidates: list[dict[str, Any]] = []
    for allocation, metadata in EXPECTED_ALLOCATIONS.items():
        rows = serving_rows(allocation, profile_inputs)
        for row in rows:
            row["model_id"] = model_id
            row["max_model_len"] = max_model_len
            row["slo"]["p95_tpot_ms"] = float(frozen["matrix"]["tpot_threshold_ms"])
        candidates.append(
            {
                "config_id": allocation,
                "kv_cache_dtype": metadata["kv_cache_dtype"],
                "state_cache_dtype": metadata["state_cache_dtype"],
                "deployment": {
                    "engine": "vllm",
                    "allocation": allocation,
                    "precision_args": canonical_precision_args(
                        str(metadata["kv_cache_dtype"]),
                        str(metadata["state_cache_dtype"]),
                    ),
                    "restart_required": True,
                },
                "capacity_profiles": [
                    {
                        "model_id": model_id,
                        "max_model_len": max_model_len,
                        "gpu_memory_utilization": utilization,
                        "cache_bytes": source(
                            f"capacity_{allocation}",
                            "cache_tensor_summary",
                            "workers",
                            "0",
                            "total_cache_bytes",
                        ),
                        "allocator_equivalent_sequence_slots": source(
                            f"capacity_{allocation}",
                            "capacity",
                            "max_concurrency",
                        ),
                    }
                ],
                "serving_profiles": rows,
                "quality_profiles": [
                    {
                        "model_id": model_id,
                        "task": "gsm8k",
                        "delta_ci95_low": source("quality_gsm8k", "candidates", allocation, "delta_ci95_low"),
                        "delta_ci95_high": source("quality_gsm8k", "candidates", allocation, "delta_ci95_high"),
                        "inference_method": source(
                            "quality_gsm8k",
                            "candidates",
                            allocation,
                            "inference_method",
                        ),
                        "estimand": source("quality_gsm8k", "candidates", allocation, "estimand"),
                        "n_seed_item_draws": source(
                            "quality_gsm8k", "candidates", allocation, "n_seed_item_draws"
                        ),
                        "n_item_clusters": source(
                            "quality_gsm8k", "candidates", allocation, "n_item_clusters"
                        ),
                        "n_seed_clusters": source(
                            "quality_gsm8k", "candidates", allocation, "n_seed_clusters"
                        ),
                        "cluster_degrees_of_freedom": source(
                            "quality_gsm8k",
                            "candidates",
                            allocation,
                            "cluster_degrees_of_freedom",
                        ),
                    }
                ],
            }
        )

    return {
        "schema_version": 1,
        "profile_status": "CALIBRATION",
        "evidence": evidence,
        "candidates": candidates,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-analysis", type=Path, required=True)
    parser.add_argument("--frozen-contract", type=Path, required=True)
    parser.add_argument("--capacity-validation", type=Path, required=True)
    parser.add_argument("--capacity-evidence", action="append", required=True)
    parser.add_argument("--quality-analysis", type=Path, required=True)
    parser.add_argument("--quality-evidence-out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    quality_path = args.quality_analysis.resolve()
    quality, quality_sha = load_verified_json(quality_path)
    quality_out = args.quality_evidence_out.resolve()
    recipe_out = args.out.resolve()
    require(not quality_out.exists(), f"refusing to overwrite quality evidence: {quality_out}")
    require(not recipe_out.exists(), f"refusing to overwrite recipe: {recipe_out}")
    normalized = normalize_quality_evidence(
        quality,
        source_path=repo_relative(quality_path, repo_root),
        source_sha256=quality_sha,
    )
    normalized_sha = atomic_write_json_with_hash(quality_out, normalized)
    recipe = build_recipe(
        calibration_path=args.calibration_analysis.resolve(),
        frozen_contract_path=args.frozen_contract.resolve(),
        capacity_validation_path=args.capacity_validation.resolve(),
        capacity_paths={key: path.resolve() for key, path in parse_capacity_paths(args.capacity_evidence).items()},
        quality_evidence_path=quality_out,
        repo_root=repo_root,
    )
    recipe_sha = atomic_write_json_with_hash(recipe_out, recipe)
    print(
        json.dumps(
            {
                "status": "BUILT",
                "quality_evidence": str(quality_out),
                "quality_evidence_sha256": normalized_sha,
                "recipe": str(recipe_out),
                "recipe_sha256": recipe_sha,
                "serving_rows": sum(len(row["serving_profiles"]) for row in recipe["candidates"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileBuildError as error:
        print(f"ERROR: {error}")
        raise SystemExit(2)
