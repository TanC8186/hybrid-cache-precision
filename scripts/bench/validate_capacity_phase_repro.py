"""Compare a capacity formal attempt with an independent reproduction attempt."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

if __package__:
    from .analyze_capacity_phase_diagram import (
        atomic_write_json,
        expected_cells,
        load_cells,
        sha256_file,
    )
else:
    from analyze_capacity_phase_diagram import (
        atomic_write_json,
        expected_cells,
        load_cells,
        sha256_file,
    )


EPSILON = 1e-12


def symmetric_relative_diff(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), EPSILON)


def evidence_outcome(comparison_passed: bool, promotion_eligible: bool) -> tuple[str, str]:
    if not comparison_passed:
        return "ANALYZED", "NOT_REPRODUCIBLE"
    if not promotion_eligible:
        return "ANALYZED", "PARTIALLY_REPRODUCIBLE"
    return "VERIFIED", "REPRODUCIBLE"


def load_frozen_contract(path: Path, attempt: str) -> tuple[dict[str, Any], str]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    digest = sha256_file(path)
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != digest:
        raise SystemExit(f"contract SHA verification failed: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_status") != "FROZEN" or contract.get("attempt_id") != attempt:
        raise SystemExit(f"contract identity mismatch: {path}")
    return contract, digest


def compare_attempts(
    parent_cells: dict[tuple[str, str, str, int, float], dict[str, Any]],
    repro_cells: dict[tuple[str, str, str, int, float], dict[str, Any]],
    token_relative_tolerance: float,
    median_gain_pp_tolerance: float,
) -> dict[str, Any]:
    required = expected_cells("formal")
    if set(parent_cells) != required or set(repro_cells) != required:
        raise SystemExit("parent or reproduction matrix differs from the formal contract")

    cell_rows = []
    token_failures = []
    for key in sorted(required):
        parent = parent_cells[key]
        repro = repro_cells[key]
        parent_tokens = int(parent["capacity"]["tokens"])
        repro_tokens = int(repro["capacity"]["tokens"])
        relative_diff = symmetric_relative_diff(parent_tokens, repro_tokens)
        row = {
            "model": key[0],
            "kv_dtype": key[1],
            "state_dtype_arg": key[2],
            "length": key[3],
            "gpu_memory_utilization": key[4],
            "parent_tokens": parent_tokens,
            "repro_tokens": repro_tokens,
            "token_symmetric_relative_diff": round(relative_diff, 8),
            "parent_num_gpu_blocks": parent["cache_config"]["num_gpu_blocks"],
            "repro_num_gpu_blocks": repro["cache_config"]["num_gpu_blocks"],
            "status": "WITHIN_TOLERANCE" if relative_diff <= token_relative_tolerance else "MISMATCH",
        }
        cell_rows.append(row)
        if relative_diff > token_relative_tolerance:
            token_failures.append(row)

    core_keys = sorted({(m, kv, length, util) for m, kv, state, length, util in required if state != "float16"})
    pair_rows = []
    grouped_parent: dict[tuple[str, str], list[float]] = {}
    grouped_repro: dict[tuple[str, str], list[float]] = {}
    direction_failures = []
    for model, kv, length, util in core_keys:
        parent_fp32 = int(parent_cells[(model, kv, "auto", length, util)]["capacity"]["tokens"])
        parent_bf16 = int(parent_cells[(model, kv, "bfloat16", length, util)]["capacity"]["tokens"])
        repro_fp32 = int(repro_cells[(model, kv, "auto", length, util)]["capacity"]["tokens"])
        repro_bf16 = int(repro_cells[(model, kv, "bfloat16", length, util)]["capacity"]["tokens"])
        parent_gain = (parent_bf16 / parent_fp32 - 1.0) * 100.0
        repro_gain = (repro_bf16 / repro_fp32 - 1.0) * 100.0
        direction_ok = parent_bf16 > parent_fp32 and repro_bf16 > repro_fp32
        row = {
            "model": model,
            "kv_dtype": kv,
            "length": length,
            "gpu_memory_utilization": util,
            "parent_gain_pct": round(parent_gain, 6),
            "repro_gain_pct": round(repro_gain, 6),
            "gain_abs_diff_pp": round(abs(parent_gain - repro_gain), 6),
            "direction_status": "MATCH" if direction_ok else "MISMATCH",
        }
        pair_rows.append(row)
        grouped_parent.setdefault((model, kv), []).append(parent_gain)
        grouped_repro.setdefault((model, kv), []).append(repro_gain)
        if not direction_ok:
            direction_failures.append(row)

    group_rows = []
    median_failures = []
    for key in sorted(grouped_parent):
        parent_median = statistics.median(grouped_parent[key])
        repro_median = statistics.median(grouped_repro[key])
        difference = abs(parent_median - repro_median)
        row = {
            "model": key[0],
            "kv_dtype": key[1],
            "parent_median_gain_pct": round(parent_median, 6),
            "repro_median_gain_pct": round(repro_median, 6),
            "absolute_diff_pp": round(difference, 6),
            "status": "WITHIN_TOLERANCE" if difference <= median_gain_pp_tolerance else "MISMATCH",
        }
        group_rows.append(row)
        if difference > median_gain_pp_tolerance:
            median_failures.append(row)

    passed = not token_failures and not direction_failures and not median_failures
    return {
        "passed": passed,
        "n_cells": len(cell_rows),
        "n_core_pairs": len(pair_rows),
        "max_token_symmetric_relative_diff": max(
            row["token_symmetric_relative_diff"] for row in cell_rows
        ),
        "token_failure_count": len(token_failures),
        "direction_failure_count": len(direction_failures),
        "median_gain_failure_count": len(median_failures),
        "cells": cell_rows,
        "core_pairs": pair_rows,
        "group_medians": group_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/verified/2026-08-11/capacity-phase"))
    parser.add_argument("--parent-attempt", required=True)
    parser.add_argument("--repro-attempt", required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--repro-contract", type=Path, required=True)
    parser.add_argument("--token-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--median-gain-pp-tolerance", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 <= args.token_relative_tolerance < 1.0:
        raise SystemExit("token relative tolerance must be in [0, 1)")
    if args.median_gain_pp_tolerance < 0.0:
        raise SystemExit("median gain tolerance must be non-negative")

    parent_contract, parent_contract_sha = load_frozen_contract(
        args.parent_contract, args.parent_attempt
    )
    repro_contract, repro_contract_sha = load_frozen_contract(args.repro_contract, args.repro_attempt)
    parent_link = repro_contract.get("parent_attempt", {})
    if parent_link.get("attempt_id") != args.parent_attempt:
        raise SystemExit("reproduction contract parent attempt mismatch")
    if parent_link.get("contract_sha256") != parent_contract_sha:
        raise SystemExit("reproduction contract parent SHA mismatch")

    tolerances = repro_contract.get("reproducibility_tolerances", {})
    if tolerances.get("cell_token_symmetric_relative") != args.token_relative_tolerance:
        raise SystemExit("CLI token tolerance differs from frozen contract")
    if tolerances.get("group_median_gain_absolute_pp") != args.median_gain_pp_tolerance:
        raise SystemExit("CLI median gain tolerance differs from frozen contract")

    parent_cells = load_cells(args.root, args.parent_attempt)
    repro_cells = load_cells(args.root, args.repro_attempt)
    comparison = compare_attempts(
        parent_cells,
        repro_cells,
        args.token_relative_tolerance,
        args.median_gain_pp_tolerance,
    )
    passed = comparison["passed"]
    promotion_eligible = bool(repro_contract.get("promotion_eligible", True))
    verification_status, verdict = evidence_outcome(passed, promotion_eligible)
    result = {
        "schema_version": 1,
        "material_passport": {
            "origin_skill": "experiment-skill",
            "origin_mode": "validate",
            "origin_date": "2026-08-11",
            "verification_status": verification_status,
            "version_label": "capacity_phase_repro_validation_v1",
        },
        "parent_attempt": args.parent_attempt,
        "reproduction_attempt": args.repro_attempt,
        "determinism_class": "environment-sensitive allocator benchmark",
        "contracts": {
            "parent_sha256": parent_contract_sha,
            "reproduction_sha256": repro_contract_sha,
        },
        "tolerances": {
            "cell_token_symmetric_relative": args.token_relative_tolerance,
            "group_median_gain_absolute_pp": args.median_gain_pp_tolerance,
            "timing_metrics_compared": False,
        },
        "promotion_gate": {
            "eligible": promotion_eligible,
            "status": "PASS" if passed and promotion_eligible else "BLOCKED",
            "reason": repro_contract.get("promotion_block_reason"),
        },
        "reproducibility_verdict": verdict,
        "comparison": comparison,
    }
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "verdict": result["reproducibility_verdict"],
                "n_cells": comparison["n_cells"],
                "n_core_pairs": comparison["n_core_pairs"],
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
