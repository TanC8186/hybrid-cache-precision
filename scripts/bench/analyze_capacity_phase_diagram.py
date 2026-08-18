"""Validate and aggregate the capacity phase-diagram sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import uuid
from pathlib import Path
from typing import Any


MODEL_PARAMS = {
    "2b": {"A_f": 6 * 2_048.0, "A_q": 6 * 528.0, "G_fp32": 18 * 1_085_440.0, "G_bf16": 18 * 561_152.0},
    "9b": {"A_f": 8 * 2_048.0, "A_q": 8 * 528.0, "G_fp32": 24 * 1_085_440.0, "G_bf16": 24 * 561_152.0},
}

CELL_RE = re.compile(
    r"^(?P<attempt>.+)__(?P<model>2b|9b)__kv(?P<kv>fp16|int4)"
    r"__state(?P<state>auto|bfloat16|float16)__L(?P<length>\d+)__u(?P<util>\d+)\.json$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest


def parse_cell_name(path: Path, attempt: str) -> dict[str, Any]:
    match = CELL_RE.match(path.name)
    if match is None or match.group("attempt") != attempt:
        raise ValueError(f"unexpected capacity cell filename: {path.name}")
    fields = match.groupdict()
    return {
        "model": fields["model"],
        "kv": fields["kv"],
        "state": fields["state"],
        "length": int(fields["length"]),
        "util": int(fields["util"]) / 100.0,
    }


def expected_cells(phase: str) -> set[tuple[str, str, str, int, float]]:
    cells: set[tuple[str, str, str, int, float]] = set()

    def add_core(model: str, lengths: list[int], utils: list[float]) -> None:
        for util in utils:
            for length in lengths:
                for kv in ("fp16", "int4"):
                    for state in ("auto", "bfloat16"):
                        cells.add((model, kv, state, length, util))

    if phase == "mvex":
        cells.update(
            {
                ("2b", "int4", "auto", 4096, 0.85),
                ("2b", "int4", "bfloat16", 4096, 0.85),
            }
        )
    elif phase == "pilot":
        add_core("2b", [1024, 4096, 16384, 32768], [0.80])
        cells.update(
            {
                ("2b", "fp16", "float16", 4096, 0.85),
                ("2b", "int4", "float16", 4096, 0.85),
            }
        )
    elif phase == "formal":
        add_core("2b", [1024, 2048, 4096, 8192, 16384, 32768], [0.70, 0.80, 0.90])
        add_core("9b", [2048, 4096, 16384, 32768], [0.80, 0.90])
        for model in ("2b", "9b"):
            for length in (4096, 16384):
                for kv in ("fp16", "int4"):
                    cells.add((model, kv, "float16", length, 0.85))
    else:
        raise ValueError("phase must be mvex, pilot, or formal")
    return cells


def predicted_state_ratio(model: str, kv: str, length: int) -> float:
    params = MODEL_PARAMS[model]
    attention_bytes = params["A_f"] if kv == "fp16" else params["A_q"]
    return (attention_bytes * length + params["G_fp32"]) / (
        attention_bytes * length + params["G_bf16"]
    )


def validate_cell_record(record: dict[str, Any], meta: dict[str, Any], path: Path) -> None:
    label = str(path)
    if record.get("schema_version") != 1 or record.get("probe") != "probe_ssm_state_dtype.py":
        raise SystemExit(f"probe schema mismatch: {label}")

    args = record.get("args", {})
    expected_state = "float32" if meta["state"] == "auto" else meta["state"]
    expected_kv_arg = "auto" if meta["kv"] == "fp16" else "int4_per_token_head"
    model_marker = "Qwen3.5-2B" if meta["model"] == "2b" else "Qwen3.5-9B"
    if model_marker not in str(args.get("model", "")):
        raise SystemExit(f"model mismatch: {label}")
    if args.get("dtype_arg") != meta["state"]:
        raise SystemExit(f"state argument mismatch: {label}")
    if record.get("resolved_mamba_ssm_cache_dtype") != expected_state:
        raise SystemExit(f"state dtype mismatch: {label}")
    if args.get("kv_cache_dtype") != expected_kv_arg:
        raise SystemExit(f"KV dtype mismatch: {label}")
    if args.get("kv_cache_dtype_per_layer") != {}:
        raise SystemExit(f"per-layer KV config must be empty: {label}")
    if args.get("seed") != 42:
        raise SystemExit(f"seed mismatch: {label}")
    if int(args.get("max_model_len", -1)) != meta["length"]:
        raise SystemExit(f"length mismatch: {label}")
    if not math.isclose(float(args.get("gpu_memory_utilization", math.nan)), meta["util"]):
        raise SystemExit(f"memory utilization mismatch: {label}")
    if record.get("generation") is not None:
        raise SystemExit(f"unexpected generation workload in capacity probe: {label}")

    capacity = record.get("capacity", {})
    tokens = capacity.get("tokens")
    concurrency = capacity.get("max_concurrency")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise SystemExit(f"invalid capacity tokens: {label}")
    if not isinstance(concurrency, (int, float)) or not math.isfinite(float(concurrency)) or concurrency <= 0:
        raise SystemExit(f"invalid max concurrency: {label}")
    if capacity.get("max_model_len") != meta["length"]:
        raise SystemExit(f"capacity max_model_len mismatch: {label}")

    cache_config = record.get("cache_config", {})
    if cache_config.get("mamba_ssm_cache_dtype") != expected_state:
        raise SystemExit(f"cache_config state dtype mismatch: {label}")
    if cache_config.get("cache_dtype") != expected_kv_arg:
        raise SystemExit(f"cache_config KV dtype mismatch: {label}")
    blocks = cache_config.get("num_gpu_blocks")
    if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks <= 0:
        raise SystemExit(f"invalid GPU block count: {label}")
    elapsed = record.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or elapsed <= 0:
        raise SystemExit(f"invalid elapsed_seconds: {label}")


def load_cells(root: Path, attempt: str) -> dict[tuple[str, str, str, int, float], dict[str, Any]]:
    cells: dict[tuple[str, str, str, int, float], dict[str, Any]] = {}
    for path in sorted(root.glob(f"{attempt}__*.json")):
        meta = parse_cell_name(path, attempt)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.is_file() or sha256_file(path) != sidecar.read_text(encoding="ascii").strip():
            raise SystemExit(f"SHA verification failed: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_cell_record(record, meta, path)
        key = (meta["model"], meta["kv"], meta["state"], meta["length"], meta["util"])
        if key in cells:
            raise SystemExit(f"duplicate capacity cell: {key}")
        cells[key] = record
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/verified/2026-08-11/capacity-phase"))
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--phase", choices=["mvex", "pilot", "formal"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cells = load_cells(args.root, args.attempt)
    required = expected_cells(args.phase)
    missing = sorted(required - set(cells))
    unexpected = sorted(set(cells) - required)
    if missing or unexpected:
        raise SystemExit(f"incomplete matrix: missing={missing} unexpected={unexpected}")

    rows = []
    residuals = []
    pair_keys = sorted({(m, kv, length, util) for m, kv, state, length, util in cells if state != "float16"})
    for model, kv, length, util in pair_keys:
        fp32 = cells[(model, kv, "auto", length, util)]
        bf16 = cells[(model, kv, "bfloat16", length, util)]
        fp32_tokens = int(fp32["capacity"]["tokens"])
        bf16_tokens = int(bf16["capacity"]["tokens"])
        measured = bf16_tokens / fp32_tokens
        predicted = predicted_state_ratio(model, kv, length)
        residual = (measured / predicted - 1.0) * 100.0
        residuals.append(residual)
        rows.append(
            {
                "model": model,
                "kv_dtype": kv,
                "length": length,
                "gpu_memory_utilization": util,
                "fp32_state_tokens": fp32_tokens,
                "bf16_state_tokens": bf16_tokens,
                "measured_state_ratio": round(measured, 6),
                "predicted_state_ratio": round(predicted, 6),
                "prediction_residual_pct": round(residual, 4),
                "fp32_allocator_equivalent_sequence_slots": fp32["capacity"]["max_concurrency"],
                "bf16_allocator_equivalent_sequence_slots": bf16["capacity"]["max_concurrency"],
                "fp32_num_gpu_blocks": fp32["cache_config"]["num_gpu_blocks"],
                "bf16_num_gpu_blocks": bf16["cache_config"]["num_gpu_blocks"],
            }
        )

    frontier = []
    for (model, kv, state, length, util), record in sorted(cells.items()):
        if state != "float16":
            continue
        frontier.append(
            {
                "model": model,
                "kv_dtype": kv,
                "state_dtype": state,
                "length": length,
                "gpu_memory_utilization": util,
                "capacity_tokens": int(record["capacity"]["tokens"]),
                "allocator_equivalent_sequence_slots": record["capacity"]["max_concurrency"],
            }
        )

    abs_residuals = [abs(value) for value in residuals]
    result = {
        "schema_version": 1,
        "material_passport": {
            "origin_skill": "experiment-skill",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "capacity_phase_analysis_v1",
        },
        "attempt": args.attempt,
        "phase": args.phase,
        "determinism_class": "deterministic_allocator_for_frozen_build_and_config",
        "model_parameters": MODEL_PARAMS,
        "capacity_semantics": (
            "allocator token capacity and L-normalized allocator-equivalent sequence slots; "
            "not demonstrated scheduler admission or SLO-serving concurrency"
        ),
        "n_cells": len(cells),
        "rows": rows,
        "float16_state_frontier": frontier,
        "prediction_residual_summary": {
            "n_pairs": len(residuals),
            "median_absolute_pct": round(statistics.median(abs_residuals), 4),
            "mean_absolute_pct": round(statistics.mean(abs_residuals), 4),
            "max_absolute_pct": round(max(abs_residuals), 4),
            "interpretation": "Residuals quantify idealized model error; they are not lower-bound evidence.",
        },
    }
    digest = atomic_write_json(args.out, result)
    print(json.dumps({"out": str(args.out), "sha256": digest, "n_cells": len(cells)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
