#!/usr/bin/env python3
"""Fail-closed integrity audit for the frozen RULER no-think pilot.

This validator is intentionally separate from the frozen statistical analyzer.
It checks provenance, exact matrix completeness, sidecar hashes, configuration
effects, sample-level schema, and score recomputation before analysis begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CELLS = [
    ("ruler_fwe", 4096, "2b"),
    ("ruler_fwe", 8192, "2b"),
    ("ruler_niah_multiquery", 4096, "9b"),
    ("ruler_niah_multiquery", 8192, "9b"),
    ("ruler_fwe", 8192, "9b"),
]
DATASET_SEEDS = (42, 11, 23)
ALLOCATIONS = ("fp16", "fp16_statebf16")
EXPECTED_SAMPLES = 20
ENGINE_SEED = 7
MAX_TOKENS = 256
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TASK_TYPES = {
    "ruler_fwe": "freq_words_extraction",
    "ruler_niah_multiquery": "niah",
}


class ValidationError(ValueError):
    """Raised when an integrity check fails."""


@dataclass(frozen=True, order=True)
class CellSpec:
    task: str
    length: int
    model: str
    dataset_seed: int
    allocation: str
    attempt: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256_sidecar(path: Path) -> str:
    sidecar = Path(str(path) + ".sha256")
    require(path.is_file(), f"missing file: {path}")
    require(sidecar.is_file(), f"missing SHA256 sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="ascii").strip().lower()
    require(bool(SHA256_RE.fullmatch(expected)), f"invalid SHA256 sidecar: {sidecar}")
    actual = sha256_file(path)
    require(actual == expected, f"SHA256 mismatch: {path}")
    return actual


def atomic_write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    digest = sha256_file(path)
    Path(str(path) + ".sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest


def cell_filename(spec: CellSpec) -> str:
    suffix = "" if spec.dataset_seed == 42 else f"__d{spec.dataset_seed}"
    return (
        f"{spec.task}__L{spec.length}__{spec.allocation}"
        f"__s{ENGINE_SEED}{suffix}.json"
    )


def expected_specs(attempt_2b: str, attempt_9b: str) -> list[CellSpec]:
    specs = []
    for task, length, model in CELLS:
        attempt = attempt_2b if model == "2b" else attempt_9b
        for dataset_seed in DATASET_SEEDS:
            for allocation in ALLOCATIONS:
                specs.append(
                    CellSpec(task, length, model, dataset_seed, allocation, attempt)
                )
    return specs


def validate_contract(root: Path, contract_path: Path, attempts: set[str]) -> tuple[dict, str]:
    contract_sha = verify_sha256_sidecar(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("schema_version") == 1, "pilot contract schema_version != 1")
    require(contract.get("classification") == "pilot_diagnostic", "unexpected pilot classification")
    require(set(contract.get("attempt_ids", [])) == attempts, "attempt IDs differ from pilot contract")

    protocol = contract.get("protocol", {})
    require(protocol.get("expected_cells") == 30, "pilot contract expected_cells != 30")
    require(protocol.get("engine_seed") == ENGINE_SEED, "pilot contract engine seed mismatch")
    require(protocol.get("max_tokens") == MAX_TOKENS, "pilot contract max_tokens mismatch")
    require(protocol.get("thinking") == "disabled", "pilot contract thinking mismatch")

    code = contract.get("code", {})
    frozen_files = {
        "runner_sha256": root / "scripts" / "eval" / "run_ruler_statebf16_nothink_5cell.sh",
        "cell_runner_sha256": root / "scripts" / "eval" / "ruler_quality.py",
        "analysis_sha256": root / "scripts" / "eval" / "analyze_ruler_nothink_5cell.py",
    }
    for field, path in frozen_files.items():
        require(path.is_file(), f"missing frozen code file: {path}")
        require(code.get(field) == sha256_file(path), f"frozen code hash mismatch: {path}")

    for relative, expected_sha in code.get("vllm_modified_file_sha256", {}).items():
        path = root / "vendor" / "vllm" / relative
        require(path.is_file(), f"missing frozen vLLM file: {path}")
        require(sha256_file(path) == expected_sha, f"frozen vLLM hash mismatch: {path}")
    return contract, contract_sha


def validate_cell_record(record: dict[str, Any], spec: CellSpec, path: Path) -> dict[str, Any]:
    label = str(path)
    require(record.get("schema_version") == 1, f"schema_version mismatch: {label}")
    require(record.get("attempt_id") == spec.attempt, f"attempt_id mismatch: {label}")
    require(record.get("status") == "completed_validated", f"status mismatch: {label}")
    require(record.get("task") == spec.task, f"task mismatch: {label}")
    require(record.get("task_type") == TASK_TYPES[spec.task], f"task_type mismatch: {label}")
    require(record.get("length") == spec.length, f"length mismatch: {label}")
    require(record.get("allocation") == spec.allocation, f"allocation mismatch: {label}")
    require(record.get("seed") == ENGINE_SEED, f"engine seed mismatch: {label}")
    require(record.get("thinking") == "disabled", f"thinking mismatch: {label}")
    require(record.get("max_tokens") == MAX_TOKENS, f"max_tokens mismatch: {label}")
    require(record.get("metric") == "string_match_all", f"metric mismatch: {label}")
    require(record.get("num_samples") == EXPECTED_SAMPLES, f"num_samples mismatch: {label}")

    sampling = record.get("sampling_params", {})
    require(sampling.get("max_tokens") == MAX_TOKENS, f"sampling max_tokens mismatch: {label}")
    require(sampling.get("thinking") == "disabled", f"sampling thinking mismatch: {label}")
    require(float(sampling.get("temperature", math.nan)) == 0.0, f"temperature mismatch: {label}")

    expected_data_suffix = f"data/ruler/{spec.task}_L{spec.length}/"
    if spec.dataset_seed != 42:
        expected_data_suffix += f"seed{spec.dataset_seed}/"
    expected_data_suffix += "validation.jsonl"
    data_file = str(record.get("data_file", "")).replace("\\", "/")
    require(data_file.endswith(expected_data_suffix), f"dataset path/seed mismatch: {label}")
    data_sha = str(record.get("data_sha256", "")).lower()
    require(bool(SHA256_RE.fullmatch(data_sha)), f"invalid data SHA256: {label}")

    effect = record.get("config_effect", {})
    detail = effect.get("detail", {})
    require(effect.get("ok") is True, f"config_effect.ok is not true: {label}")
    require(effect.get("allocation") == spec.allocation, f"config effect allocation mismatch: {label}")
    require(detail.get("cache_dtype") in {"auto", "fp16", "bf16"}, f"KV dtype mismatch: {label}")
    require(not detail.get("per_layer"), f"unexpected per-layer KV config: {label}")
    require(detail.get("a2_flag") is False, f"unexpected page-group flag: {label}")
    expected_state = "bfloat16" if spec.allocation == "fp16_statebf16" else "float32"
    require(detail.get("mamba_ssm_cache_dtype") == expected_state, f"state dtype mismatch: {label}")

    engine = record.get("engine", {})
    kwargs = engine.get("kwargs", {})
    model_marker = "Qwen3.5-2B" if spec.model == "2b" else "Qwen3.5-9B"
    require(model_marker in str(engine.get("model", "")), f"model mismatch: {label}")
    require(kwargs.get("seed") == ENGINE_SEED, f"engine kwargs seed mismatch: {label}")
    require(kwargs.get("max_model_len") == 16384, f"max_model_len mismatch: {label}")
    require(math.isclose(float(kwargs.get("gpu_memory_utilization", math.nan)), 0.85), f"GPU util mismatch: {label}")
    require(kwargs.get("enforce_eager") is True, f"enforce_eager mismatch: {label}")
    require(kwargs.get("disable_log_stats") is True, f"disable_log_stats mismatch: {label}")
    if spec.allocation == "fp16_statebf16":
        require(kwargs.get("mamba_ssm_cache_dtype") == "bfloat16", f"missing bf16 state arg: {label}")
    else:
        require("mamba_ssm_cache_dtype" not in kwargs, f"baseline has state override: {label}")

    cases = record.get("cases")
    require(isinstance(cases, list) and len(cases) == EXPECTED_SAMPLES, f"case count mismatch: {label}")
    indices = []
    total_hits = 0
    total_refs = 0
    case_signature = []
    for case in cases:
        index = case.get("index")
        references = case.get("references")
        hits = case.get("hits")
        require(isinstance(index, int) and not isinstance(index, bool), f"invalid case index: {label}")
        require(isinstance(references, list) and references, f"invalid references: {label}")
        require(isinstance(hits, list) and len(hits) == len(references), f"invalid hits: {label}")
        require(all(isinstance(hit, bool) for hit in hits), f"non-boolean hit: {label}")
        prompt_tokens = case.get("prompt_tokens")
        output_tokens = case.get("output_tokens")
        require(isinstance(prompt_tokens, int) and prompt_tokens > 0, f"invalid prompt_tokens: {label}")
        require(isinstance(output_tokens, int) and 0 <= output_tokens <= MAX_TOKENS, f"invalid output_tokens: {label}")
        indices.append(index)
        total_hits += sum(hits)
        total_refs += len(hits)
        case_signature.append((index, tuple(references), prompt_tokens))
    require(len(set(indices)) == EXPECTED_SAMPLES, f"duplicate case indices: {label}")

    accuracy = float(record.get("accuracy", math.nan))
    require(math.isfinite(accuracy) and 0.0 <= accuracy <= 100.0, f"invalid accuracy: {label}")
    recomputed_accuracy = round(100.0 * total_hits / total_refs, 2)
    require(math.isclose(accuracy, recomputed_accuracy, abs_tol=0.011), f"accuracy recomputation mismatch: {label}")
    elapsed = float(record.get("elapsed_s", math.nan))
    require(math.isfinite(elapsed) and elapsed > 0.0, f"invalid elapsed_s: {label}")

    return {
        "spec": asdict(spec),
        "data_relative_path": expected_data_suffix,
        "data_sha256": data_sha,
        "accuracy": accuracy,
        "recomputed_accuracy": recomputed_accuracy,
        "state_dtype": expected_state,
        "ruler_commit": record.get("ruler_commit"),
        "vllm_version": engine.get("vllm_version"),
        "host": record.get("host"),
        "case_signature": case_signature,
    }


def validate_matrix(
    ruler_dir: Path,
    specs: list[CellSpec],
    contract: dict[str, Any],
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    if repo_root is None:
        repo_root = ruler_dir.parents[2]
    attempts = sorted({spec.attempt for spec in specs})
    for attempt in attempts:
        attempt_dir = ruler_dir / attempt
        require(attempt_dir.is_dir(), f"missing attempt directory: {attempt_dir}")
        expected_json = {attempt_dir / cell_filename(spec) for spec in specs if spec.attempt == attempt}
        actual_json = set(attempt_dir.glob("*.json"))
        require(actual_json == expected_json, f"incomplete or unexpected JSON matrix in {attempt_dir}")
        expected_sidecars = {Path(str(path) + ".sha256") for path in expected_json}
        actual_sidecars = set(attempt_dir.glob("*.json.sha256"))
        require(actual_sidecars == expected_sidecars, f"incomplete or unexpected SHA matrix in {attempt_dir}")

    summaries = []
    by_pair: dict[tuple[str, int, str, int], dict[str, dict[str, Any]]] = {}
    data_by_dataset: dict[tuple[str, int, int], set[str]] = {}
    for spec in specs:
        path = ruler_dir / spec.attempt / cell_filename(spec)
        cell_sha = verify_sha256_sidecar(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        summary = validate_cell_record(record, spec, path)
        source_path = repo_root / summary["data_relative_path"]
        require(source_path.is_file(), f"missing source dataset: {source_path}")
        require(sha256_file(source_path) == summary["data_sha256"], f"source dataset SHA mismatch: {source_path}")
        source_rows = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        require(len(source_rows) == EXPECTED_SAMPLES, f"source dataset row count mismatch: {source_path}")
        source_signature = [
            (row.get("index"), tuple(row.get("outputs", [])))
            for row in source_rows
        ]
        result_signature = [
            (index, references)
            for index, references, _prompt_tokens in summary["case_signature"]
        ]
        require(source_signature == result_signature, f"result/source case mismatch: {path}")
        summary["path"] = str(path)
        summary["sha256"] = cell_sha
        summaries.append(summary)
        pair_key = (spec.task, spec.length, spec.model, spec.dataset_seed)
        by_pair.setdefault(pair_key, {})[spec.allocation] = summary
        data_key = (spec.task, spec.length, spec.dataset_seed)
        data_by_dataset.setdefault(data_key, set()).add(summary["data_sha256"])

    for key, pair in by_pair.items():
        require(set(pair) == set(ALLOCATIONS), f"allocation pair incomplete: {key}")
        baseline = pair["fp16"]
        treatment = pair["fp16_statebf16"]
        require(baseline["data_sha256"] == treatment["data_sha256"], f"paired data SHA mismatch: {key}")
        require(baseline["case_signature"] == treatment["case_signature"], f"paired cases mismatch: {key}")
    for key, hashes in data_by_dataset.items():
        require(len(hashes) == 1, f"cross-model dataset SHA mismatch: {key}")

    expected_host = contract.get("host", {}).get("hostname")
    expected_vllm = contract.get("code", {}).get("vllm_version")
    require({row["host"] for row in summaries} == {expected_host}, "host provenance mismatch")
    require({row["vllm_version"] for row in summaries} == {expected_vllm}, "vLLM version mismatch")
    require(len({row["ruler_commit"] for row in summaries}) == 1, "RULER commit mismatch across cells")
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruler-dir", type=Path, default=Path("results/quality/ruler-subset"))
    parser.add_argument("--attempt-2b", default="ruler-statebf16-nothink-20260811-2b")
    parser.add_argument("--attempt-9b", default="ruler-statebf16-nothink-20260811-9b")
    parser.add_argument(
        "--pilot-contract",
        type=Path,
        default=Path("results/quality/ruler-nothink-5cell-pilot-20260811.contract.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/quality/ruler-nothink-5cell-validation-20260811.json"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    ruler_dir = args.ruler_dir if args.ruler_dir.is_absolute() else root / args.ruler_dir
    contract_path = args.pilot_contract if args.pilot_contract.is_absolute() else root / args.pilot_contract
    out = args.out if args.out.is_absolute() else root / args.out
    specs = expected_specs(args.attempt_2b, args.attempt_9b)
    attempts = {args.attempt_2b, args.attempt_9b}

    try:
        contract, contract_sha = validate_contract(root, contract_path, attempts)
        summaries = validate_matrix(ruler_dir, specs, contract, root)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise SystemExit(f"RULER integrity validation failed: {exc}") from exc

    compact_cells = []
    for row in summaries:
        compact = {key: value for key, value in row.items() if key != "case_signature"}
        compact_cells.append(compact)
    result = {
        "schema_version": 1,
        "material_passport": {
            "origin_skill": "experiment-skill",
            "origin_mode": "validate",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "ANALYZED",
            "version_label": "ruler_nothink_integrity_v1",
        },
        "validation_id": "ruler-nothink-5cell-integrity-20260811",
        "gate": "Gate 2 pilot integrity",
        "gate_status": "PASS",
        "evidence_status": "UNVERIFIED",
        "reproducibility_rerun": "not_run",
        "expected_cells": len(specs),
        "validated_cells": len(summaries),
        "attempts": sorted(attempts),
        "pilot_contract": {"path": str(contract_path), "sha256": contract_sha},
        "validator_sha256": sha256_file(Path(__file__)),
        "checks": [
            "exact 30-cell matrix and adjacent SHA256 sidecars",
            "frozen runner, cell runner, analyzer, and patched vLLM file hashes",
            "status, task, length, allocation, engine seed, dataset seed, no-think, and token budget",
            "config_effect.ok and exact fp32/bf16 recurrent-state dtype",
            "20 sample records per cell and accuracy recomputed from sample hits",
            "paired allocation dataset identity and cross-model dataset identity",
            "frozen host and vLLM provenance",
        ],
        "cells": compact_cells,
    }
    digest = atomic_write_json(out, result)
    print(json.dumps({"out": str(out), "sha256": digest, "validated_cells": len(summaries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
