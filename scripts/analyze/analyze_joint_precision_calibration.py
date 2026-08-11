"""Audit and aggregate a four-allocation joint-precision calibration attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.analyze.verify_a2_reproduction import (
    VerificationError,
    load_json,
    require,
    sha256_file,
    utc_timestamp,
    verify_sidecar,
    write_json_with_hash,
)
from scripts.analyze.verify_joint_precision_mvex import (
    EXPECTED_ALLOCATIONS,
    audit_sample,
    audit_server,
    finite,
)
from scripts.bench.run_steady_state import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def canonical_json_sha(value: Any, *, sort_keys: bool) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=sort_keys,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def student_t_summary(values: Sequence[float], t_critical: float) -> dict[str, Any]:
    parsed = [finite("repeat metric", value) for value in values]
    require(len(parsed) >= 2, "at least two independent repeats are required")
    critical = finite("t critical", t_critical)
    require(critical > 0, "t critical must be positive")
    mean = statistics.fmean(parsed)
    sample_sd = statistics.stdev(parsed)
    margin = critical * sample_sd / math.sqrt(len(parsed))
    return {
        "values": parsed,
        "n": len(parsed),
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def aggregate_calibration(
    samples: Sequence[Mapping[str, Any]],
    frozen_matrix: Mapping[str, Any],
    *,
    t_critical: float,
) -> dict[str, Any]:
    groups: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (
            str(sample["allocation"]),
            str(sample["workload"]),
            float(sample["offered_rate_req_s"]),
        )
        groups[key].append(sample)

    allocations = [str(row["id"]) for row in frozen_matrix["allocations"]]
    seeds = sorted(int(seed) for seed in frozen_matrix["seeds"])
    expected_keys = {
        (allocation, workload, float(rate))
        for allocation in allocations
        for workload, rates in frozen_matrix["workload_rates_req_s"].items()
        for rate in rates
    }
    require(set(groups) == expected_keys, "observed allocation-workload-rate cells do not match the frozen matrix")

    thresholds = [f"{float(value):g}" for value in frozen_matrix["ttft_thresholds_ms"]]
    profile_inputs: dict[str, dict[str, dict[str, Any]]] = {allocation: {} for allocation in allocations}
    cells: list[dict[str, Any]] = []
    for allocation, workload, rate in sorted(expected_keys):
        repeats = sorted(groups[(allocation, workload, rate)], key=lambda row: int(row["seed"]))
        observed_seeds = [int(row["seed"]) for row in repeats]
        require(observed_seeds == seeds, f"{allocation}/{workload}/{rate:g}: repeat seeds do not match")
        require(len({row["sample_id"] for row in repeats}) == len(seeds), "duplicate sample ID in a repeat cell")

        throughput = student_t_summary(
            [float(row["request_throughput_req_s"]) for row in repeats],
            t_critical,
        )
        ttft = student_t_summary([float(row["ttft_p95_ms"]) for row in repeats], t_critical)
        tpot = student_t_summary([float(row["tpot_p95_ms"]) for row in repeats], t_critical)
        slo_sweep: dict[str, Any] = {}
        for threshold in thresholds:
            goodput = student_t_summary(
                [float(row["slo_sweep"][threshold]["goodput_req_s"]) for row in repeats],
                t_critical,
            )
            slo_sweep[threshold] = {
                "ttft_threshold_ms": float(threshold),
                "tpot_threshold_ms": float(frozen_matrix["tpot_threshold_ms"]),
                "goodput_req_s": goodput,
                "profile_eligible": goodput["ci95_low"] > 0,
            }

        expected_requests = sum(int(row["accounting"]["expected"]) for row in repeats)
        completed_requests = sum(int(row["accounting"]["completed"]) for row in repeats)
        failed_requests = sum(int(row["accounting"]["failed"]) for row in repeats)
        require(completed_requests + failed_requests == expected_requests, "cell request denominator drift")
        rate_key = f"{rate:g}"
        profile_input = {
            "offered_rate_req_s": rate,
            "n_independent_repeats": len(repeats),
            "p95_ttft_ucb_ms": ttft["ci95_high"],
            "p95_tpot_ucb_ms": tpot["ci95_high"],
            "slo_sweep": {
                threshold: {
                    "slo_goodput_lcb_req_s": value["goodput_req_s"]["ci95_low"],
                    "profile_eligible": value["profile_eligible"],
                }
                for threshold, value in slo_sweep.items()
            },
        }
        profile_inputs[allocation].setdefault(workload, {})[rate_key] = profile_input
        cells.append(
            {
                "allocation": allocation,
                "workload": workload,
                "offered_rate_req_s": rate,
                "sample_ids": [str(row["sample_id"]) for row in repeats],
                "seeds": observed_seeds,
                "expected_requests": expected_requests,
                "completed_requests": completed_requests,
                "failed_requests": failed_requests,
                "request_throughput_req_s": throughput,
                "p95_ttft_ms": ttft,
                "p95_tpot_ms": tpot,
                "slo_sweep": slo_sweep,
            }
        )

    return {
        "cell_count": len(cells),
        "profile_row_count": len(cells) * len(thresholds),
        "cells": cells,
        "profile_inputs": profile_inputs,
    }


def calibration_fallacy_scan() -> list[dict[str, str]]:
    return [
        {
            "fallacy": "Simpson's paradox",
            "severity": "NOTE",
            "detail": "Results remain stratified by allocation, workload, and offered rate; no pooled direction is used.",
        },
        {
            "fallacy": "Ecological fallacy",
            "severity": "NOTE",
            "detail": "Inference is limited to seed/trace repeats; requests are not treated as independent repeats.",
        },
        {
            "fallacy": "Berkson's paradox",
            "severity": "CAUTION",
            "detail": "The single GPU, model, and predeclared rate grid are selected deployment conditions.",
        },
        {
            "fallacy": "Collider bias",
            "severity": "NOTE",
            "detail": "No post-treatment covariate adjustment or conditioned regression is performed.",
        },
        {
            "fallacy": "Base rate neglect",
            "severity": "NOTE",
            "detail": "Every rate reports the full request denominator and failed-request fraction.",
        },
        {
            "fallacy": "Regression to the mean",
            "severity": "NOTE",
            "detail": "Rates and seeds were frozen before execution rather than selected from extreme outcomes.",
        },
        {
            "fallacy": "Survivorship bias",
            "severity": "NOTE",
            "detail": "The audit requires all 144 samples and retains every failed request as an SLO miss.",
        },
        {
            "fallacy": "Look-elsewhere effect",
            "severity": "CAUTION",
            "detail": "The 240 pointwise SLO profile rows are calibration inputs, not claim-level significance tests.",
        },
        {
            "fallacy": "Garden of forking paths",
            "severity": "NOTE",
            "detail": "Seeds, rates, thresholds, t critical, and aggregation rules are frozen in the contract.",
        },
        {
            "fallacy": "Correlation != causation",
            "severity": "NOTE",
            "detail": "Allocation is experimentally controlled; claims remain bounded to this deployment matrix.",
        },
        {
            "fallacy": "Reverse causality",
            "severity": "NOTE",
            "detail": "Precision configuration precedes each serving measurement through a cold-start deployment epoch.",
        },
    ]


def validate_attempt(
    attempt_dir: Path,
    frozen_contract_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    frozen_sha = verify_sidecar(frozen_contract_path.with_suffix(frozen_contract_path.suffix + ".sha256"))
    frozen = load_json(frozen_contract_path)
    attempt_contract_sha = verify_sidecar(attempt_dir / "attempt_contract.json.sha256")
    environment_sha = verify_sidecar(attempt_dir / "environment.json.sha256")
    summary_sha = verify_sidecar(attempt_dir / "summary.json.sha256")
    attempt_contract = load_json(attempt_dir / "attempt_contract.json")
    environment = load_json(attempt_dir / "environment.json")
    summary = load_json(attempt_dir / "summary.json")

    require(frozen["contract_status"] == "FROZEN", "outer contract is not frozen")
    require(attempt_contract["attempt_id"] == frozen["attempt_id"], "attempt ID drift")
    require(attempt_contract["parent_attempt"] == frozen["parent_attempt"], "parent attempt drift")
    require(attempt_contract["phase"]["name"] == "calibration", "attempt phase is not calibration")
    require(attempt_contract["config_sha256"] == frozen["code"]["config_sha256"], "config hash drift")
    require(environment["root_git"]["clean"] is True, "captured root worktree was dirty")
    require(environment["root_git"]["status"] == "", "captured root status was not empty")
    require(environment["root_git"]["commit"] == attempt_contract["git_commit"], "root commit drift")
    require(environment["vllm_source_commit"] == frozen["host"]["vllm_commit"], "vLLM commit drift")
    require(environment["model_config_sha256"] == frozen["model"]["config_sha256"], "model config drift")
    sharegpt = environment["datasets"].get("sharegpt")
    require(isinstance(sharegpt, dict), "ShareGPT dataset provenance is missing")
    require(sharegpt["sha256"] == frozen["datasets"]["sharegpt"]["sha256"], "ShareGPT hash drift")
    require(sharegpt["size_bytes"] == frozen["datasets"]["sharegpt"]["size_bytes"], "ShareGPT size drift")

    plan = attempt_contract["plan"]
    matrix = frozen["matrix"]
    require(len(plan) == matrix["expected_samples"], "frozen sample denominator drift")
    require(canonical_json_sha(plan, sort_keys=True) == matrix["plan_sha256"], "plan digest drift")
    sample_ids = [str(row["sample_id"]) for row in plan]
    require(canonical_json_sha(sample_ids, sort_keys=False) == matrix["sample_ids_sha256"], "sample ID digest drift")
    require(
        sum(int(row["num_prompts"]) for row in plan) == matrix["expected_measurement_requests"], "request plan drift"
    )
    require(summary["counts"] == {"completed_validated": matrix["expected_samples"]}, "summary is incomplete")
    require(
        [row["sample_id"] for row in summary["samples"]] == sample_ids,
        "summary sample ordering or membership drift",
    )
    require(all(row["status"] == "completed_validated" for row in summary["samples"]), "summary status drift")
    sample_root = attempt_dir / "samples"
    observed_sample_dirs = {path.name for path in sample_root.iterdir() if path.is_dir()}
    require(observed_sample_dirs == set(sample_ids), "sample directory set differs from the frozen denominator")

    config_path = repo_root / frozen["code"]["config_path"]
    require(sha256_file(config_path) == frozen["code"]["config_sha256"], "auditor config hash drift")
    config = load_config(config_path)
    root_commit = str(attempt_contract["git_commit"])
    vllm_commit = str(attempt_contract["vllm_source_commit"])
    samples = [
        audit_sample(
            attempt_dir,
            row,
            root_commit=root_commit,
            vllm_commit=vllm_commit,
            require_zero_failures=False,
        )
        for row in plan
    ]
    servers = [
        audit_server(attempt_dir, allocation, config["allocations"][allocation]) for allocation in EXPECTED_ALLOCATIONS
    ]
    aggregated = aggregate_calibration(
        samples,
        matrix,
        t_critical=float(frozen["statistical_plan"]["t_critical_df2"]),
    )

    expected_requests = sum(int(row["accounting"]["expected"]) for row in samples)
    completed_requests = sum(int(row["accounting"]["completed"]) for row in samples)
    failed_requests = sum(int(row["accounting"]["failed"]) for row in samples)
    require(expected_requests == matrix["expected_measurement_requests"], "audited request denominator drift")
    require(completed_requests + failed_requests == expected_requests, "audited request conservation failed")
    fallacies = calibration_fallacy_scan()
    require(len(fallacies) == 11, "fallacy scan must cover 11 categories")
    return {
        "schema_version": 1,
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "origin_date": utc_timestamp()[:10],
            "verification_status": "ANALYZED",
            "version_label": "joint_precision_calibration_analysis_v1",
        },
        "gate": "PASS",
        "evidence_status": "ANALYZED",
        "attempt_id": frozen["attempt_id"],
        "parent_attempt": frozen["parent_attempt"],
        "root_commit": root_commit,
        "vllm_commit": vllm_commit,
        "provenance": {
            "frozen_contract_path": frozen_contract_path.relative_to(repo_root).as_posix(),
            "frozen_contract_sha256": frozen_sha,
            "attempt_contract_sha256": attempt_contract_sha,
            "environment_sha256": environment_sha,
            "summary_sha256": summary_sha,
            "sharegpt_sha256": sharegpt["sha256"],
        },
        "completeness": {
            "expected_samples": matrix["expected_samples"],
            "audited_samples": len(samples),
            "expected_measurement_requests": matrix["expected_measurement_requests"],
            "completed_requests": completed_requests,
            "failed_requests": failed_requests,
            "request_conservation": completed_requests + failed_requests == expected_requests,
            "server_sessions": len(servers),
            "silent_exclusions": 0,
        },
        "statistical_method": frozen["statistical_plan"],
        "aggregation": aggregated,
        "servers": servers,
        "fallacy_scan": {
            "coverage": "11/11",
            "items": fallacies,
        },
        "promotion": {
            "profile_construction_authorized": True,
            "confirmatory_run_authorized": True,
            "paper_quantitative_use_authorized": False,
            "reason": "Calibration passed completeness and analysis gates but uses tuning seeds and has no reproducibility rerun.",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--frozen-contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    report = validate_attempt(
        args.attempt_dir.resolve(),
        args.frozen_contract.resolve(),
        repo_root=repo_root,
    )
    digest = write_json_with_hash(args.out.resolve(), report)
    print(json.dumps({"status": "ANALYZED", "out": str(args.out.resolve()), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"ERROR: {error}")
        raise SystemExit(2)
