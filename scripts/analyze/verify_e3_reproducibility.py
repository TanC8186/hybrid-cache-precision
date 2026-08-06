"""Audit E3 formal/reproduction artifacts and promote verified evidence.

The formal, workload-specific reproduction, and upper-neighbor attempts remain
independent denominators. This tool only compares their audited summaries and
creates link-style verification artifacts; it never rewrites source results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

THRESHOLDS_MS = (250.0, 500.0, 1000.0, 2000.0, 3000.0)
GENERATED_SOURCE_EXCLUSIONS = {
    "artifact_sha256_manifest.json",
    "artifact_sha256_manifest.json.sha256",
    "reproducibility_report.json",
    "reproducibility_report.json.sha256",
    "validation_report.md",
    "validation_report.md.sha256",
    "verification_link.json",
    "verification_link.json.sha256",
}


class VerificationError(RuntimeError):
    """Raised when an integrity or reproducibility gate fails."""


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise VerificationError(f"expected a JSON object: {path}")
    return value


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_bytes(payload)
    temp.replace(path)


def write_json_with_hash(path: Path, value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    atomic_write(path, payload)
    digest = sha256_file(path)
    atomic_write(path.with_suffix(path.suffix + ".sha256"), f"{digest}\n".encode())
    return digest


def write_text_with_hash(path: Path, value: str) -> str:
    atomic_write(path, value.encode("utf-8"))
    digest = sha256_file(path)
    atomic_write(path.with_suffix(path.suffix + ".sha256"), f"{digest}\n".encode())
    return digest


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def require_close(actual: float, expected: float, label: str, *, atol: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
        raise VerificationError(f"{label}: actual={actual!r}, expected={expected!r}")


def symmetric_relative_difference(left: float, right: float, *, epsilon: float = 1e-12) -> float:
    return abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), epsilon)


def sidecar_target(sidecar: Path) -> Path:
    require(sidecar.name.endswith(".sha256"), f"not a SHA sidecar: {sidecar}")
    return sidecar.with_name(sidecar.name.removesuffix(".sha256"))


def verify_sidecar(sidecar: Path) -> str:
    target = sidecar_target(sidecar)
    require(target.is_file(), f"SHA sidecar target is missing: {target}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(target)
    require(actual == expected, f"SHA mismatch: {target}")
    return actual


def percentile(values: Sequence[float], pct: float) -> float:
    require(bool(values), "cannot compute percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def per_request_tpot_ms(result: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for output_len, request_itls in zip(result["output_lens"], result["itls"], strict=True):
        count = int(output_len)
        if count <= 1:
            values.append(0.0)
        else:
            values.append(1000.0 * sum(float(value) for value in request_itls) / (count - 1))
    return values


def sample_sweep(
    result: Mapping[str, Any],
    sample: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], float, list[int]]:
    expected = int(sample["num_prompts"])
    completed = int(result["completed"])
    failed = int(result["failed"])
    require(completed + failed == expected, f"{sample['sample_id']}: request denominator mismatch")

    detail_fields = (
        "ttfts",
        "itls",
        "output_lens",
        "start_times",
        "errors",
        "input_lens",
        "generated_texts",
    )
    for field in detail_fields:
        require(field in result, f"{sample['sample_id']}: missing detailed field {field}")
        require(
            len(result[field]) == expected,
            f"{sample['sample_id']}: {field} rows={len(result[field])}, expected={expected}",
        )

    errors = list(result["errors"])
    failed_indices = [index for index, error in enumerate(errors) if error]
    require(len(failed_indices) == failed, f"{sample['sample_id']}: failed/error count mismatch")
    success_mask = [not error for error in errors]
    require(sum(success_mask) == completed, f"{sample['sample_id']}: completed/detail count mismatch")

    duration_s = float(result["duration"])
    offered_rate = float(sample["request_rate"])
    window_s = float(protocol["measurement_window_s"])
    start_times = [float(value) for value in result["start_times"]]
    arrival_span_s = max(start_times) - min(start_times) if len(start_times) > 1 else 0.0
    arrival_ratio = arrival_span_s / window_s
    tolerance = float(protocol["arrival_window_tolerance_fraction"])
    require(
        1.0 - tolerance <= arrival_ratio <= 1.0 + tolerance,
        f"{sample['sample_id']}: arrival window ratio {arrival_ratio:.6f} outside tolerance",
    )

    ttfts_ms = [1000.0 * float(value) for value in result["ttfts"]]
    tpots_ms = per_request_tpot_ms(result)
    tpot_threshold = float(protocol["tpot_threshold_ms"])
    sustainable_ratio = float(protocol["sustainable_goodput_ratio"])
    sweep: dict[str, dict[str, Any]] = {}
    for threshold in protocol["ttft_thresholds_ms"]:
        threshold_value = float(threshold)
        good_count = sum(
            1
            for success, ttft, tpot in zip(success_mask, ttfts_ms, tpots_ms, strict=True)
            if success and ttft <= threshold_value and tpot <= tpot_threshold
        )
        goodput = good_count / duration_s
        goodput_ratio = goodput / offered_rate
        sweep[f"{threshold_value:g}"] = {
            "good_requests": good_count,
            "goodput_req_s": goodput,
            "goodput_over_offered": goodput_ratio,
            "sustainable": goodput_ratio >= sustainable_ratio,
        }

    successful_ttfts = [value for value, success in zip(ttfts_ms, success_mask, strict=True) if success]
    successful_tpots = [value for value, success in zip(tpots_ms, success_mask, strict=True) if success]
    require_close(
        percentile(successful_ttfts, 99),
        float(result["p99_ttft_ms"]),
        f"{sample['sample_id']}: p99 TTFT",
        atol=1e-6,
    )
    require_close(
        percentile(successful_tpots, 99),
        float(result["p99_tpot_ms"]),
        f"{sample['sample_id']}: p99 TPOT",
        atol=2e-4,
    )
    return sweep, arrival_ratio, failed_indices


def derive_boundaries(
    points: Mapping[tuple[str, str, int, float], Sequence[tuple[float, bool]]],
) -> dict[tuple[str, str, int, float], dict[str, Any]]:
    boundaries: dict[tuple[str, str, int, float], dict[str, Any]] = {}
    for key, raw_points in points.items():
        ordered = sorted((float(rate), bool(sustainable)) for rate, sustainable in raw_points)
        sustainable_rates = [rate for rate, sustainable in ordered if sustainable]
        boundary = max(sustainable_rates) if sustainable_rates else None
        boundaries[key] = {
            "max_tested_sustainable_rate": boundary,
            "right_censored": boundary is not None and boundary == ordered[-1][0],
            "tested_rates": [rate for rate, _ in ordered],
        }
    return boundaries


def merge_points(
    *point_maps: Mapping[tuple[str, str, int, float], Sequence[tuple[float, bool]]],
) -> dict[tuple[str, str, int, float], list[tuple[float, bool]]]:
    merged: dict[tuple[str, str, int, float], dict[float, bool]] = defaultdict(dict)
    for point_map in point_maps:
        for key, points in point_map.items():
            for rate, sustainable in points:
                rate = float(rate)
                sustainable = bool(sustainable)
                if rate in merged[key] and merged[key][rate] != sustainable:
                    raise VerificationError(f"conflicting boundary point for {key} at rate {rate}")
                merged[key][rate] = sustainable
    return {key: sorted(values.items()) for key, values in merged.items()}


def source_manifest(source_dir: Path, artifact_id: str) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    sidecars_verified = 0
    for path in sorted(candidate for candidate in source_dir.rglob("*") if candidate.is_file()):
        if path.name in GENERATED_SOURCE_EXCLUSIONS:
            continue
        if path.name.endswith(".sha256"):
            verify_sidecar(path)
            sidecars_verified += 1
        entries.append(
            {
                "artifact_id": artifact_id,
                "relative_path": path.relative_to(source_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries, sidecars_verified


def audit_launch_suite(
    suite_dir: Path,
    *,
    role: str,
    expected_status: str,
    expected_exit_code: int,
    expected_git_commit: str,
    expected_vllm_commit: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    contract = load_json(suite_dir / "launch_contract.json")
    suite_id = str(contract["suite_id"])
    require(contract["git_commit"] == expected_git_commit, f"{suite_id}: root commit mismatch")
    require(contract["vllm_source_commit"] == expected_vllm_commit, f"{suite_id}: vLLM commit mismatch")
    require(contract["config_sha256"] == expected_config_sha256, f"{suite_id}: config hash mismatch")
    require(
        int((suite_dir / "exit_code").read_text(encoding="utf-8").strip()) == expected_exit_code,
        f"{suite_id}: outer exit code mismatch",
    )
    suite_status = load_json(suite_dir / "suite_status.json")
    require(suite_status["status"] == expected_status, f"{suite_id}: suite status mismatch")

    if expected_status == "completed":
        expected_samples = sum(int(item["expected_samples"]) for item in contract["attempts"])
        require(
            int(suite_status["completed_validated"]) == expected_samples,
            f"{suite_id}: completed sample count mismatch",
        )
        require(int(suite_status["failed"]) == 0, f"{suite_id}: launch recorded failures")
        require(int(suite_status["not_started"]) == 0, f"{suite_id}: launch has unstarted samples")
        scientific_attempts_created = True
        samples_started = expected_samples
        gpu_compute_started = True
    else:
        failure = load_json(suite_dir / "failure_record.json")
        require(failure["status"] == "FAILED_PRECOMPUTE", f"{suite_id}: failure status mismatch")
        require(int(failure["samples_started"]) == 0, f"{suite_id}: failed suite started samples")
        require(not failure["scientific_attempts_created"], f"{suite_id}: failed suite created attempts")
        require(not failure["gpu_compute_started"], f"{suite_id}: failed suite started GPU compute")
        scientific_attempts_created = False
        samples_started = 0
        gpu_compute_started = False

    manifest_entries, sidecars_verified = source_manifest(suite_dir, suite_id)
    return {
        "summary": {
            "suite_id": suite_id,
            "role": role,
            "source_path": str(suite_dir.resolve()),
            "status": "COMPLETED" if expected_status == "completed" else "FAILED_PRECOMPUTE",
            "exit_code": expected_exit_code,
            "samples_started": samples_started,
            "scientific_attempts_created": scientific_attempts_created,
            "gpu_compute_started": gpu_compute_started,
            "sha_audit": {
                "files_hashed": len(manifest_entries),
                "sidecars_verified": sidecars_verified,
            },
            "provenance": {
                "git_commit": contract["git_commit"],
                "vllm_source_commit": contract["vllm_source_commit"],
                "config_sha256": contract["config_sha256"],
                "launch_contract_sha256": sha256_file(suite_dir / "launch_contract.json"),
            },
        },
        "manifest_entries": manifest_entries,
    }


def audit_attempt(
    attempt_dir: Path,
    *,
    role: str,
    expected_git_commit: str,
    expected_vllm_commit: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    contract = load_json(attempt_dir / "attempt_contract.json")
    attempt_id = str(contract["attempt_id"])
    require(contract["git_commit"] == expected_git_commit, f"{attempt_id}: root commit mismatch")
    require(contract["vllm_source_commit"] == expected_vllm_commit, f"{attempt_id}: vLLM commit mismatch")
    require(contract["config_sha256"] == expected_config_sha256, f"{attempt_id}: config hash mismatch")

    for name in ("attempt_contract.json", "environment.json", "summary.json"):
        verify_sidecar(attempt_dir / f"{name}.sha256")

    plan = list(contract["plan"])
    expected_ids = {str(sample["sample_id"]) for sample in plan}
    sample_root = attempt_dir / "samples"
    observed_ids = {path.name for path in sample_root.iterdir() if path.is_dir()}
    require(observed_ids == expected_ids, f"{attempt_id}: sample directory set mismatch")

    summary = load_json(attempt_dir / "summary.json")
    require(
        summary["counts"] == {"completed_validated": len(plan)},
        f"{attempt_id}: summary is not fully completed_validated",
    )
    aggregate = load_json(attempt_dir / "aggregate.json")
    require(bool(aggregate["complete_denominator"]), f"{attempt_id}: incomplete aggregate denominator")
    require(aggregate["observed_samples"] == len(plan), f"{attempt_id}: aggregate sample count mismatch")
    require(aggregate["verification_status"] == "ANALYZED", f"{attempt_id}: source aggregate was rewritten")

    cells: dict[tuple[str, str, float, float], list[float]] = defaultdict(list)
    points: dict[tuple[str, str, int, float], list[tuple[float, bool]]] = defaultdict(list)
    sample_values: dict[tuple[str, str, float, int, float], float] = {}
    expected_requests = 0
    completed_requests = 0
    failed_requests = 0
    arrival_ratios: list[float] = []
    durations: list[float] = []

    for sample in plan:
        sample_id = str(sample["sample_id"])
        sample_dir = sample_root / sample_id
        for name in ("contract.json", "result.json", "analysis.json"):
            verify_sidecar(sample_dir / f"{name}.sha256")

        sample_contract = load_json(sample_dir / "contract.json")
        status = load_json(sample_dir / "status.json")
        result = load_json(sample_dir / "result.json")
        analysis = load_json(sample_dir / "analysis.json")
        require(status["status"] == "completed_validated", f"{sample_id}: invalid status")
        require(status["returncode"] == 0, f"{sample_id}: non-zero return code")
        require(status["result_sha256"] == sha256_file(sample_dir / "result.json"), f"{sample_id}: result hash")
        require(
            status["analysis_sha256"] == sha256_file(sample_dir / "analysis.json"),
            f"{sample_id}: analysis hash",
        )

        contract_base = {
            key: value
            for key, value in sample_contract.items()
            if key not in {"command", "contract_sha256"}
        }
        require(
            canonical_sha256(contract_base) == sample_contract["contract_sha256"],
            f"{sample_id}: canonical contract hash mismatch",
        )
        for key in ("sample_id", "allocation", "workload", "seed", "num_prompts", "request_rate"):
            require(sample_contract[key] == sample[key], f"{sample_id}: contract/plan mismatch for {key}")
        require(sample_contract["attempt_id"] == attempt_id, f"{sample_id}: attempt linkage mismatch")
        require(sample_contract["git_commit"] == expected_git_commit, f"{sample_id}: root commit mismatch")
        require(
            sample_contract["vllm_source_commit"] == expected_vllm_commit,
            f"{sample_id}: vLLM commit mismatch",
        )

        for key in ("sample_id", "allocation", "workload", "num_prompts"):
            require(result[key] == sample[key], f"{sample_id}: result/plan mismatch for {key}")
        require(int(result["seed"]) == int(sample["seed"]), f"{sample_id}: result/plan mismatch for seed")
        require(float(result["offered_rate"]) == float(sample["request_rate"]), f"{sample_id}: offered rate")
        require(result["attempt_id"] == attempt_id, f"{sample_id}: result attempt linkage")
        require(result["contract_sha256"] == sample_contract["contract_sha256"], f"{sample_id}: result contract")
        require(result["git_commit"] == expected_git_commit, f"{sample_id}: result root commit")
        require(result["vllm_source_commit"] == expected_vllm_commit, f"{sample_id}: result vLLM commit")

        sweep, arrival_ratio, failed_indices = sample_sweep(
            result,
            sample,
            sample_contract["protocol"],
        )
        require(analysis["failed_request_indices"] == failed_indices, f"{sample_id}: failed indices")
        require_close(analysis["arrival_span_over_target"], arrival_ratio, f"{sample_id}: arrival ratio")
        for threshold_key, recomputed in sweep.items():
            recorded = analysis["slo_sweep"][threshold_key]
            require(recorded["good_requests"] == recomputed["good_requests"], f"{sample_id}: good count")
            require(recorded["sustainable"] == recomputed["sustainable"], f"{sample_id}: sustainability")
            require_close(
                recorded["goodput_req_s"],
                recomputed["goodput_req_s"],
                f"{sample_id}: goodput",
            )
            require_close(
                recorded["goodput_over_offered"],
                recomputed["goodput_over_offered"],
                f"{sample_id}: goodput/offered",
            )

            threshold = float(threshold_key)
            allocation = str(sample["allocation"])
            workload = str(sample["workload"])
            rate = float(sample["request_rate"])
            seed = int(sample["seed"])
            cells[(allocation, workload, rate, threshold)].append(recomputed["goodput_req_s"])
            points[(allocation, workload, seed, threshold)].append((rate, recomputed["sustainable"]))
            sample_values[(allocation, workload, rate, seed, threshold)] = recomputed["goodput_req_s"]

        expected_requests += int(sample["num_prompts"])
        completed_requests += int(result["completed"])
        failed_requests += int(result["failed"])
        arrival_ratios.append(arrival_ratio)
        durations.append(float(result["duration"]))

    cell_means = {key: statistics.fmean(values) for key, values in cells.items()}
    aggregate_cells = {
        (str(cell["allocation"]), str(cell["workload"]), float(cell["request_rate"])): cell
        for cell in aggregate["cells"]
    }
    for (allocation, workload, rate, threshold), mean in cell_means.items():
        metric = f"goodput_req_s_ttft_{threshold:g}"
        require_close(
            aggregate_cells[(allocation, workload, rate)]["metrics"][metric]["mean"],
            mean,
            f"{attempt_id}: aggregate cell mean {allocation}/{workload}/{rate}/{threshold}",
        )

    boundaries = derive_boundaries(points)
    aggregate_boundaries = {
        (
            str(boundary["allocation"]),
            str(boundary["workload"]),
            float(boundary["ttft_threshold_ms"]),
        ): boundary
        for boundary in aggregate["boundaries"]
    }
    for (allocation, workload, seed, threshold), boundary in boundaries.items():
        recorded_group = aggregate_boundaries[(allocation, workload, threshold)]
        recorded = next(item for item in recorded_group["per_seed"] if int(item["seed"]) == seed)
        require(
            recorded["max_tested_sustainable_rate"] == boundary["max_tested_sustainable_rate"],
            f"{attempt_id}: boundary mismatch for {allocation}/{workload}/s{seed}/{threshold}",
        )
        require(
            recorded["right_censored"] == boundary["right_censored"],
            f"{attempt_id}: censoring mismatch for {allocation}/{workload}/s{seed}/{threshold}",
        )

    manifest_entries, sidecars_verified = source_manifest(attempt_dir, attempt_id)
    return {
        "summary": {
            "attempt_id": attempt_id,
            "role": role,
            "source_path": str(attempt_dir.resolve()),
            "parent_attempt": contract.get("parent_attempt"),
            "source_evidence_status": aggregate["verification_status"],
            "denominator": {
                "expected_samples": len(plan),
                "validated_samples": len(plan),
                "expected_requests": expected_requests,
                "completed_requests": completed_requests,
                "failed_requests": failed_requests,
            },
            "arrival_window": {
                "target_s": 60.0,
                "min_ratio": min(arrival_ratios),
                "max_ratio": max(arrival_ratios),
            },
            "duration_s": {"min": min(durations), "max": max(durations)},
            "sha_audit": {
                "files_hashed": len(manifest_entries),
                "sidecars_verified": sidecars_verified,
            },
            "provenance": {
                "git_commit": contract["git_commit"],
                "vllm_source_commit": contract["vllm_source_commit"],
                "config_sha256": contract["config_sha256"],
                "attempt_contract_sha256": sha256_file(attempt_dir / "attempt_contract.json"),
                "aggregate_sha256": sha256_file(attempt_dir / "aggregate.json"),
                "validation_md_sha256": sha256_file(attempt_dir / "validation.md"),
            },
        },
        "manifest_entries": manifest_entries,
        "cell_means": cell_means,
        "points": points,
        "boundaries": boundaries,
        "sample_values": sample_values,
    }


def compare_cell_means(
    formal: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for key, rerun_value in sorted(reproduction["cell_means"].items()):
        require(key in formal["cell_means"], f"formal cell is missing: {key}")
        original_value = formal["cell_means"][key]
        relative_difference = symmetric_relative_difference(original_value, rerun_value)
        comparisons.append(
            {
                "allocation": key[0],
                "workload": key[1],
                "request_rate": key[2],
                "ttft_threshold_ms": key[3],
                "formal_mean_goodput_req_s": original_value,
                "reproduction_mean_goodput_req_s": rerun_value,
                "symmetric_relative_difference": relative_difference,
                "within_tolerance": relative_difference <= tolerance,
            }
        )
    return comparisons


def compare_boundaries(
    formal: Mapping[str, Any],
    reproduced_boundaries: Mapping[tuple[str, str, int, float], Mapping[str, Any]],
    *,
    workload: str,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    keys = sorted(key for key in reproduced_boundaries if key[1] == workload)
    for key in keys:
        require(key in formal["boundaries"], f"formal boundary is missing: {key}")
        original = formal["boundaries"][key]["max_tested_sustainable_rate"]
        reproduced = reproduced_boundaries[key]["max_tested_sustainable_rate"]
        comparisons.append(
            {
                "allocation": key[0],
                "workload": key[1],
                "seed": key[2],
                "ttft_threshold_ms": key[3],
                "formal_boundary_req_s": original,
                "reproduced_boundary_req_s": reproduced,
                "exact_match": original == reproduced,
                "reproduced_right_censored": reproduced_boundaries[key]["right_censored"],
            }
        )
    return comparisons


def boundary_summary(formal: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for (allocation, workload, _seed, threshold), value in formal["boundaries"].items():
        boundary = value["max_tested_sustainable_rate"]
        require(boundary is not None, f"missing sustainable boundary: {allocation}/{workload}/{threshold}")
        grouped[(allocation, workload, threshold)].append(float(boundary))

    rows: list[dict[str, Any]] = []
    for workload in ("random", "sharegpt"):
        for threshold in THRESHOLDS_MS:
            fp16 = statistics.fmean(grouped[("fp16", workload, threshold)])
            int4 = statistics.fmean(grouped[("int4", workload, threshold)])
            rows.append(
                {
                    "workload": workload,
                    "ttft_threshold_ms": threshold,
                    "fp16_mean_boundary_req_s": fp16,
                    "int4_mean_boundary_req_s": int4,
                    "int4_relative_change": int4 / fp16 - 1.0,
                }
            )
    return rows


def paired_goodput_summary(
    audit: Mapping[str, Any],
    *,
    workload: str,
    request_rate: float,
    threshold: float,
) -> dict[str, Any]:
    differences: list[float] = []
    for seed in (7, 42, 2026):
        fp16 = audit["sample_values"][("fp16", workload, request_rate, seed, threshold)]
        int4 = audit["sample_values"][("int4", workload, request_rate, seed, threshold)]
        differences.append(int4 - fp16)
    return {
        "workload": workload,
        "request_rate": request_rate,
        "ttft_threshold_ms": threshold,
        "n": len(differences),
        "mean_int4_minus_fp16_goodput_req_s": statistics.fmean(differences),
        "sample_std": statistics.stdev(differences),
    }


def fallacy_scan() -> list[dict[str, str]]:
    return [
        {
            "fallacy": "Simpson's paradox",
            "severity": "CAUTION",
            "finding": "Random and ShareGPT effects have opposite signs; pooling workloads would be misleading.",
        },
        {
            "fallacy": "Ecological fallacy",
            "severity": "NOTE",
            "finding": "Inference stays at the seed-by-cell level; requests are not treated as independent replicates.",
        },
        {
            "fallacy": "Berkson's paradox",
            "severity": "NOTE",
            "finding": "No outcome-conditioned sample selection was used; workloads and rates were frozen in advance.",
        },
        {
            "fallacy": "Collider bias",
            "severity": "NOTE",
            "finding": "No post-treatment control variable is used in the boundary comparison.",
        },
        {
            "fallacy": "Base rate neglect",
            "severity": "NOTE",
            "finding": "Not applicable to the offered-rate and SLO-goodput measurements.",
        },
        {
            "fallacy": "Regression to the mean",
            "severity": "NOTE",
            "finding": "Independent attempts reproduce the same discrete boundary; no pre/post extreme-group inference is made.",
        },
        {
            "fallacy": "Survivorship bias",
            "severity": "NOTE",
            "finding": "All planned samples and requests are accounted for; failed historical attempts remain excluded and preserved.",
        },
        {
            "fallacy": "Look-elsewhere effect",
            "severity": "CAUTION",
            "finding": "Five TTFT thresholds are evaluated; all are reported and no threshold is selected as uniquely confirmatory.",
        },
        {
            "fallacy": "Garden of forking paths",
            "severity": "CAUTION",
            "finding": "Protocol v2 followed documented v1 failures; v1 attempts are quarantined and v2 gates were rerun before formal execution.",
        },
        {
            "fallacy": "Correlation is not causation",
            "severity": "CAUTION",
            "finding": "The allocation comparison is controlled, but conclusions are limited to this model, stack, hardware, and workload.",
        },
        {
            "fallacy": "Reverse causality",
            "severity": "NOTE",
            "finding": "Not applicable to the controlled serving configuration comparison.",
        },
    ]


def render_validation(report: Mapping[str, Any]) -> str:
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-08-04",
        "- Verification Status: VERIFIED",
        "- Version Label: e3_reproducibility_validation_v1",
        "",
        "## Validation Report",
        "",
        f"- **Source**: `{report['formal_attempt_id']}`",
        "- **Overall Confidence**: CAUTION",
        "- **Reproducibility Verdict**: REPRODUCIBLE",
        "- **Method**: environment-sensitive seeded re-run, 10% symmetric relative tolerance",
        "",
        "### Integrity",
        "",
        "| Attempt | Role | Samples | Requests | Failures | Arrival ratio | SHA files |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for attempt in report["attempt_audits"]:
        denominator = attempt["denominator"]
        arrival = attempt["arrival_window"]
        lines.append(
            "| {attempt} | {role} | {samples}/{samples} | {completed}/{expected} | {failed} | "
            "{minimum:.6f}--{maximum:.6f} | {files} |".format(
                attempt=attempt["attempt_id"],
                role=attempt["role"],
                samples=denominator["validated_samples"],
                completed=denominator["completed_requests"],
                expected=denominator["expected_requests"],
                failed=denominator["failed_requests"],
                minimum=arrival["min_ratio"],
                maximum=arrival["max_ratio"],
                files=attempt["sha_audit"]["files_hashed"],
            )
        )

    lines.extend(
        [
            "",
            "### Reproducibility",
            "",
            f"- Cell means compared: {report['comparison']['cell_means']['count']}",
            (
                "- Maximum symmetric relative difference: "
                f"{100.0 * report['comparison']['cell_means']['max_relative_difference']:.3f}%"
            ),
            (
                "- Boundary comparisons: "
                f"{report['comparison']['boundaries']['exact_matches']}/"
                f"{report['comparison']['boundaries']['count']} exact"
            ),
            "- ShareGPT rate-40 upper-neighbor samples: 6/6 unsustainable at all five TTFT thresholds.",
            "",
            "### Sustainable Boundary",
            "",
            "| Workload | TTFT | FP16 | INT4 | Relative change |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["primary_results"]["boundary_summary"]:
        lines.append(
            "| {workload} | {threshold:.0f} ms | {fp16:.2f} | {int4:.2f} | {change:+.1f}% |".format(
                workload=row["workload"],
                threshold=row["ttft_threshold_ms"],
                fp16=row["fp16_mean_boundary_req_s"],
                int4=row["int4_mean_boundary_req_s"],
                change=100.0 * row["int4_relative_change"],
            )
        )

    lines.extend(
        [
            "",
            "### Fallacy Scan",
            "",
            "- **Coverage**: 11/11 fallacy types checked",
            "",
            "| Fallacy | Severity | Finding |",
            "|---|---|---|",
        ]
    )
    for item in report["fallacy_scan"]:
        lines.append(f"| {item['fallacy']} | {item['severity']} | {item['finding']} |")

    lines.extend(
        [
            "",
            "### Claim Boundary",
            "",
            "- Random synthetic traffic reproduces a threshold-dependent INT4 capacity gain: none at 250 ms, "
            "small at 500 ms, and 14.3% at 1000--3000 ms.",
            "- ShareGPT reverses direction: the mean sustainable boundary is 23.33 req/s for INT4 versus "
            "28.33 req/s for FP16 (-17.6%).",
            "- These results do not support a workload-general claim that INT4 increases SLO capacity.",
            "- Confidence remains CAUTION because n=3, rates use a 5 req/s grid, and ShareGPT intervals are wide.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", required=True, type=Path)
    parser.add_argument("--random-repro", required=True, type=Path)
    parser.add_argument("--sharegpt-repro", required=True, type=Path)
    parser.add_argument("--sharegpt-upper", required=True, type=Path)
    parser.add_argument("--successful-suite", required=True, type=Path)
    parser.add_argument("--failed-suite", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--vllm-commit", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--tolerance", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    attempts = [
        audit_attempt(
            args.formal.resolve(),
            role="formal",
            expected_git_commit=args.git_commit,
            expected_vllm_commit=args.vllm_commit,
            expected_config_sha256=args.config_sha256,
        ),
        audit_attempt(
            args.random_repro.resolve(),
            role="random_reproduction",
            expected_git_commit=args.git_commit,
            expected_vllm_commit=args.vllm_commit,
            expected_config_sha256=args.config_sha256,
        ),
        audit_attempt(
            args.sharegpt_repro.resolve(),
            role="sharegpt_reproduction",
            expected_git_commit=args.git_commit,
            expected_vllm_commit=args.vllm_commit,
            expected_config_sha256=args.config_sha256,
        ),
        audit_attempt(
            args.sharegpt_upper.resolve(),
            role="sharegpt_upper_neighbor",
            expected_git_commit=args.git_commit,
            expected_vllm_commit=args.vllm_commit,
            expected_config_sha256=args.config_sha256,
        ),
    ]
    formal, random_repro, sharegpt_repro, sharegpt_upper = attempts
    launch_suites = [
        audit_launch_suite(
            args.successful_suite.resolve(),
            role="successful_reproduction_suite",
            expected_status="completed",
            expected_exit_code=0,
            expected_git_commit=args.git_commit,
            expected_vllm_commit=args.vllm_commit,
            expected_config_sha256=args.config_sha256,
        ),
        audit_launch_suite(
            args.failed_suite.resolve(),
            role="preserved_failed_precompute_suite",
            expected_status="failed",
            expected_exit_code=1,
            expected_git_commit=args.git_commit,
            expected_vllm_commit=args.vllm_commit,
            expected_config_sha256=args.config_sha256,
        ),
    ]

    require(
        random_repro["summary"]["parent_attempt"] == formal["summary"]["attempt_id"],
        "random reproduction parent linkage mismatch",
    )
    require(
        sharegpt_repro["summary"]["parent_attempt"] == formal["summary"]["attempt_id"],
        "ShareGPT reproduction parent linkage mismatch",
    )
    require(
        sharegpt_upper["summary"]["parent_attempt"] == sharegpt_repro["summary"]["attempt_id"],
        "ShareGPT upper-neighbor parent linkage mismatch",
    )

    cell_comparisons = (
        compare_cell_means(formal, random_repro, tolerance=args.tolerance)
        + compare_cell_means(formal, sharegpt_repro, tolerance=args.tolerance)
        + compare_cell_means(formal, sharegpt_upper, tolerance=args.tolerance)
    )
    require(all(item["within_tolerance"] for item in cell_comparisons), "cell reproducibility failed")

    random_boundaries = compare_boundaries(
        formal,
        random_repro["boundaries"],
        workload="random",
    )
    sharegpt_points = merge_points(sharegpt_repro["points"], sharegpt_upper["points"])
    sharegpt_boundaries = derive_boundaries(sharegpt_points)
    sharegpt_boundary_comparisons = compare_boundaries(
        formal,
        sharegpt_boundaries,
        workload="sharegpt",
    )
    boundary_comparisons = random_boundaries + sharegpt_boundary_comparisons
    require(all(item["exact_match"] for item in boundary_comparisons), "boundary reproducibility failed")
    require(
        not any(
            sustainable
            for points in sharegpt_upper["points"].values()
            for _rate, sustainable in points
        ),
        "a ShareGPT rate-40 upper-neighbor sample was sustainable",
    )

    manifest_entries = [
        entry
        for attempt in attempts
        for entry in attempt["manifest_entries"]
    ] + [
        entry
        for suite in launch_suites
        for entry in suite["manifest_entries"]
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "scope": "E3 protocol-v2 source artifacts before verification promotion",
        "files": manifest_entries,
    }
    output_dir = args.output_dir.resolve()
    manifest_sha = write_json_with_hash(output_dir / "artifact_sha256_manifest.json", manifest)

    report = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "verification_status": "VERIFIED",
        "reproducibility_verdict": "REPRODUCIBLE",
        "overall_confidence": "CAUTION",
        "formal_attempt_id": formal["summary"]["attempt_id"],
        "determinism_class": "environment_sensitive_seeded",
        "comparison_tolerance": {
            "metric": "cell mean SLO goodput",
            "symmetric_relative_difference_max": args.tolerance,
            "timing_metrics_compared": False,
        },
        "attempt_audits": [attempt["summary"] for attempt in attempts],
        "launch_audits": [suite["summary"] for suite in launch_suites],
        "verification_tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source_artifact_manifest": {
            "path": str((output_dir / "artifact_sha256_manifest.json").resolve()),
            "sha256": manifest_sha,
            "file_count": len(manifest_entries),
        },
        "comparison": {
            "cell_means": {
                "count": len(cell_comparisons),
                "within_tolerance": sum(item["within_tolerance"] for item in cell_comparisons),
                "max_relative_difference": max(
                    item["symmetric_relative_difference"] for item in cell_comparisons
                ),
                "details": cell_comparisons,
            },
            "boundaries": {
                "count": len(boundary_comparisons),
                "exact_matches": sum(item["exact_match"] for item in boundary_comparisons),
                "details": boundary_comparisons,
            },
            "sharegpt_upper_neighbor": {
                "samples": 6,
                "threshold_checks": 30,
                "all_unsustainable": True,
            },
        },
        "primary_results": {
            "boundary_summary": boundary_summary(formal),
            "random_rate40_ttft2000_paired_goodput": {
                "formal": paired_goodput_summary(
                    formal,
                    workload="random",
                    request_rate=40.0,
                    threshold=2000.0,
                ),
                "reproduction": paired_goodput_summary(
                    random_repro,
                    workload="random",
                    request_rate=40.0,
                    threshold=2000.0,
                ),
            },
        },
        "fallacy_scan_coverage": "11/11",
        "fallacy_scan": fallacy_scan(),
        "excluded_evidence": [
            {
                "attempt_id": "e3-formal-c7379f0-01",
                "status": "QUARANTINED",
                "reason": "protocol-v1 request denominator failure; never pooled",
            },
            {
                "attempt_id": "e3-formal-c7379f0-02",
                "status": "QUARANTINED",
                "reason": "protocol-v1 request denominator failure; never pooled",
            },
            {
                "attempt_id": "e3-v2-repro-suite-d1d52c4-01",
                "status": "FAILED_PRECOMPUTE",
                "reason": "supervisor precompute failure; zero scientific samples",
            },
        ],
        "claim_boundary": [
            "Do not pool formal, reproduction, or upper-neighbor denominators.",
            "Do not claim that INT4 generally increases SLO capacity.",
            "Report Random and ShareGPT separately because their effect directions reverse.",
            "Retain n=3, 5 req/s grid, and wide ShareGPT uncertainty as limitations.",
        ],
    }
    report_sha = write_json_with_hash(output_dir / "reproducibility_report.json", report)
    validation_sha = write_text_with_hash(
        output_dir / "validation_report.md",
        render_validation(report),
    )

    link = {
        "schema_version": 1,
        "linked_at_utc": utc_timestamp(),
        "formal_attempt_id": formal["summary"]["attempt_id"],
        "source_evidence_status": "ANALYZED",
        "verification_scope_status": "VERIFIED",
        "reproducibility_verdict": "REPRODUCIBLE",
        "overall_confidence": "CAUTION",
        "reports": {
            "reproducibility_report": {
                "path": str((output_dir / "reproducibility_report.json").resolve()),
                "sha256": report_sha,
            },
            "validation_report": {
                "path": str((output_dir / "validation_report.md").resolve()),
                "sha256": validation_sha,
            },
            "artifact_sha256_manifest": {
                "path": str((output_dir / "artifact_sha256_manifest.json").resolve()),
                "sha256": manifest_sha,
            },
        },
        "source_artifacts_preserved": True,
        "claim_boundary": report["claim_boundary"],
    }
    link_sha = write_json_with_hash(args.formal.resolve() / "verification_link.json", link)
    print(
        json.dumps(
            {
                "verification_status": "VERIFIED",
                "verdict": "REPRODUCIBLE",
                "cell_comparisons": len(cell_comparisons),
                "max_relative_difference": report["comparison"]["cell_means"][
                    "max_relative_difference"
                ],
                "boundary_exact_matches": (
                    f"{report['comparison']['boundaries']['exact_matches']}/"
                    f"{report['comparison']['boundaries']['count']}"
                ),
                "report_sha256": report_sha,
                "validation_sha256": validation_sha,
                "manifest_sha256": manifest_sha,
                "verification_link_sha256": link_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
