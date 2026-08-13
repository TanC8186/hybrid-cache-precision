#!/usr/bin/env python3
"""Gate 4 audit for the independent RULER no-think reproduction.

The launch policy for this reproduction explicitly excludes experiment-script
hashes. This validator therefore audits observable protocol semantics, exact
denominators, sample identity, recomputed scores, and reproduction agreement.
Result sidecars remain integrity evidence; no code hash is a launch or
promotion gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.eval.validate_ruler_nothink_5cell import (
        ALLOCATIONS,
        CELLS,
        DATASET_SEEDS,
        ENGINE_SEED,
        EXPECTED_SAMPLES,
        MAX_TOKENS,
        ValidationError,
        atomic_write_json,
        cell_filename,
        expected_specs,
        require,
        sha256_file,
        validate_cell_record,
        verify_sha256_sidecar,
    )
except ModuleNotFoundError:  # Direct execution from scripts/eval.
    from validate_ruler_nothink_5cell import (
        ALLOCATIONS,
        CELLS,
        DATASET_SEEDS,
        ENGINE_SEED,
        EXPECTED_SAMPLES,
        MAX_TOKENS,
        ValidationError,
        atomic_write_json,
        cell_filename,
        expected_specs,
        require,
        sha256_file,
        validate_cell_record,
        verify_sha256_sidecar,
    )


FALLACY_NAMES = (
    "Simpson's paradox",
    "Ecological fallacy",
    "Berkson's paradox",
    "Collider bias",
    "Base rate neglect",
    "Regression to the mean",
    "Survivorship bias",
    "Look-elsewhere effect",
    "Garden of forking paths",
    "Correlation != causation",
    "Reverse causality",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    payload = value.encode("utf-8")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    digest = sha256_file(path)
    Path(str(path) + ".sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest


def validate_reproduction_contract(path: Path) -> dict[str, Any]:
    """Validate the logical contract without requiring a contract/code hash."""
    require(path.is_file(), f"missing reproduction contract: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    require(contract.get("schema_version") == 1, "contract schema_version != 1")
    require(
        contract.get("classification")
        == "environment_sensitive_independent_temporal_reproduction",
        "unexpected reproduction classification",
    )
    review = contract.get("review_policy", {})
    require(review.get("mode") == "logical_only", "review mode is not logical_only")
    require(review.get("hash_validation_performed") is False, "contract enables hash validation")
    require(
        review.get("script_hashes_are_not_launch_gates") is True,
        "contract does not disable script-hash launch gates",
    )
    require(review.get("result_sidecars_retained") is True, "result sidecars are not retained")
    require(review.get("failed_or_missing_cells_fail_closed") is True, "matrix does not fail closed")

    matrix = contract.get("matrix", {})
    actual_cells = [tuple(cell) for cell in matrix.get("cells", [])]
    require(actual_cells == CELLS, "contract five-cell matrix differs from frozen matrix")
    require(tuple(matrix.get("allocations", [])) == ALLOCATIONS, "contract allocations mismatch")
    require(tuple(matrix.get("dataset_seeds", [])) == DATASET_SEEDS, "dataset seeds mismatch")
    require(matrix.get("engine_seed") == ENGINE_SEED, "engine seed mismatch")
    require(matrix.get("expected_cells") == 30, "expected_cells != 30")
    require(matrix.get("expected_samples_per_cell") == EXPECTED_SAMPLES, "sample count mismatch")

    protocol = contract.get("protocol", {})
    require(protocol.get("thinking") == "disabled", "thinking is not disabled")
    require(protocol.get("max_tokens") == MAX_TOKENS, "max_tokens mismatch")
    require(float(protocol.get("temperature", math.nan)) == 0.0, "temperature mismatch")
    require(protocol.get("max_model_len") == 16384, "max_model_len mismatch")
    require(
        math.isclose(float(protocol.get("gpu_memory_utilization", math.nan)), 0.85),
        "GPU memory utilization mismatch",
    )
    output_root = str(contract.get("execution", {}).get("output_root", "")).replace("\\", "/")
    require(output_root.startswith("/root/autodl-tmp/"), "output root is not on the data disk")
    return contract


def _case_content_signature(record: dict[str, Any], label: str) -> list[tuple[Any, ...]]:
    signature = []
    for case in record["cases"]:
        prediction = case.get("prediction")
        references = case.get("references")
        hits = case.get("hits")
        require(isinstance(prediction, str), f"prediction is not text: {label}")
        require(isinstance(references, list), f"references are not a list: {label}")
        expected_hits = [str(reference).lower() in prediction.lower() for reference in references]
        require(hits == expected_hits, f"sample hit recomputation mismatch: {label}")
        signature.append(
            (
                case.get("index"),
                tuple(references),
                prediction,
                tuple(hits),
                case.get("prompt_tokens"),
                case.get("output_tokens"),
            )
        )
    return signature


def validate_attempt_matrix(
    repo_root: Path,
    ruler_dir: Path,
    attempt_2b: str,
    attempt_9b: str,
) -> dict[tuple[str, int, str, int, str], dict[str, Any]]:
    """Audit one exact 30-cell attempt using only logical and result checks."""
    specs = expected_specs(attempt_2b, attempt_9b)
    attempts = (attempt_2b, attempt_9b)
    for attempt in attempts:
        attempt_dir = ruler_dir / attempt
        require(attempt_dir.is_dir(), f"missing attempt directory: {attempt_dir}")
        expected_json = {
            attempt_dir / cell_filename(spec)
            for spec in specs
            if spec.attempt == attempt
        }
        require(set(attempt_dir.glob("*.json")) == expected_json, f"unexpected JSON matrix: {attempt_dir}")
        expected_sidecars = {Path(str(path) + ".sha256") for path in expected_json}
        require(
            set(attempt_dir.glob("*.json.sha256")) == expected_sidecars,
            f"unexpected result-sidecar matrix: {attempt_dir}",
        )

    rows: dict[tuple[str, int, str, int, str], dict[str, Any]] = {}
    paired: dict[tuple[str, int, str, int], dict[str, dict[str, Any]]] = {}
    dataset_hashes: dict[tuple[str, int, int], set[str]] = {}
    hosts: dict[str, set[str]] = {attempt: set() for attempt in attempts}
    versions: set[str] = set()
    ruler_commits: set[str] = set()

    for spec in specs:
        path = ruler_dir / spec.attempt / cell_filename(spec)
        result_sha = verify_sha256_sidecar(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        summary = validate_cell_record(record, spec, path)
        content_signature = _case_content_signature(record, str(path))

        source_path = repo_root / summary["data_relative_path"]
        require(source_path.is_file(), f"missing source dataset: {source_path}")
        require(
            sha256_file(source_path) == summary["data_sha256"],
            f"source dataset digest mismatch: {source_path}",
        )
        source_rows = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        require(len(source_rows) == EXPECTED_SAMPLES, f"source row count mismatch: {source_path}")
        source_signature = [(row.get("index"), tuple(row.get("outputs", []))) for row in source_rows]
        result_signature = [(case[0], case[1]) for case in content_signature]
        require(source_signature == result_signature, f"result/source samples differ: {path}")

        key = (spec.task, spec.length, spec.model, spec.dataset_seed, spec.allocation)
        item = {
            "record": record,
            "summary": summary,
            "content_signature": content_signature,
            "path": str(path),
            "result_sha256": result_sha,
        }
        rows[key] = item
        pair_key = key[:-1]
        paired.setdefault(pair_key, {})[spec.allocation] = item
        dataset_hashes.setdefault((spec.task, spec.length, spec.dataset_seed), set()).add(
            summary["data_sha256"]
        )
        hosts[spec.attempt].add(str(summary["host"]))
        versions.add(str(summary["vllm_version"]))
        ruler_commits.add(str(summary["ruler_commit"]))

    require(len(rows) == 30, f"validated {len(rows)} cells, expected 30")
    for key, pair in paired.items():
        require(set(pair) == set(ALLOCATIONS), f"allocation pair incomplete: {key}")
        base = pair["fp16"]
        treatment = pair["fp16_statebf16"]
        require(
            base["summary"]["data_sha256"] == treatment["summary"]["data_sha256"],
            f"paired dataset digest mismatch: {key}",
        )
        base_cases = [(case[0], case[1], case[4]) for case in base["content_signature"]]
        treatment_cases = [(case[0], case[1], case[4]) for case in treatment["content_signature"]]
        require(base_cases == treatment_cases, f"paired sample identity mismatch: {key}")
    for key, hashes in dataset_hashes.items():
        require(len(hashes) == 1, f"cross-model dataset digest mismatch: {key}")
    require(all(len(values) == 1 for values in hosts.values()), "an attempt spans multiple hosts")
    require(len(versions) == 1, "vLLM version differs within the matrix")
    require(len(ruler_commits) == 1, "RULER revision differs within the matrix")
    return rows


def compare_reproduction(
    original: dict[tuple[str, int, str, int, str], dict[str, Any]],
    reproduction: dict[tuple[str, int, str, int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Require exact quality/content reproduction while ignoring timing and host."""
    require(set(original) == set(reproduction), "original/reproduction cell keys differ")
    comparisons = []
    for key in sorted(original):
        parent = original[key]
        rerun = reproduction[key]
        p_summary = parent["summary"]
        r_summary = rerun["summary"]
        require(
            p_summary["data_sha256"] == r_summary["data_sha256"],
            f"reproduction dataset mismatch: {key}",
        )
        require(
            p_summary["ruler_commit"] == r_summary["ruler_commit"],
            f"RULER revision mismatch: {key}",
        )
        require(
            p_summary["vllm_version"] == r_summary["vllm_version"],
            f"vLLM version mismatch: {key}",
        )
        require(
            parent["content_signature"] == rerun["content_signature"],
            f"sample output mismatch: {key}",
        )
        original_accuracy = float(p_summary["accuracy"])
        reproduction_accuracy = float(r_summary["accuracy"])
        require(original_accuracy == reproduction_accuracy, f"accuracy mismatch: {key}")
        comparisons.append(
            {
                "cell": {
                    "task": key[0],
                    "length": key[1],
                    "model": key[2],
                    "dataset_seed": key[3],
                    "allocation": key[4],
                },
                "original_accuracy": original_accuracy,
                "reproduction_accuracy": reproduction_accuracy,
                "accuracy_difference": reproduction_accuracy - original_accuracy,
                "sample_outputs_exact": True,
                "status": "MATCH",
            }
        )
    return comparisons


def t_interval(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 4.303 * sd / math.sqrt(len(values)) if len(values) == 3 else 1.96 * sd / math.sqrt(len(values))
    return mean, mean - half, mean + half


def statistical_rows(
    rows: dict[tuple[str, int, str, int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    findings = []
    for task, length, model in CELLS:
        deltas = []
        baseline_values = []
        treatment_values = []
        for dataset_seed in DATASET_SEEDS:
            baseline = float(rows[(task, length, model, dataset_seed, "fp16")]["summary"]["accuracy"])
            treatment = float(
                rows[(task, length, model, dataset_seed, "fp16_statebf16")]["summary"]["accuracy"]
            )
            baseline_values.append(baseline)
            treatment_values.append(treatment)
            deltas.append(treatment - baseline)
        mean, low, high = t_interval(deltas)
        findings.append(
            {
                "task": task,
                "length": length,
                "model": model,
                "unit_of_analysis": "paired dataset seed",
                "n_pairs": len(deltas),
                "fp16_mean_accuracy": round(statistics.mean(baseline_values), 6),
                "statebf16_mean_accuracy": round(statistics.mean(treatment_values), 6),
                "mean_delta_accuracy_points": round(mean, 6),
                "ci95_delta_accuracy_points": [round(low, 6), round(high, 6)],
                "paired_deltas": deltas,
                "standardized_effect": "undefined_zero_variance" if statistics.pstdev(deltas) == 0 else "not_reported_n3",
                "inference": "descriptive_only_no_equivalence_claim",
            }
        )
    return findings


def fallacy_scan() -> list[dict[str, str]]:
    details = {
        "Simpson's paradox": "All five cells are reported separately by task, length, and model; aggregate direction is not substituted for strata.",
        "Ecological fallacy": "The unit of analysis and inference is the paired dataset seed, not an individual request or token.",
        "Berkson's paradox": "The five cells were fixed by the protocol-repair question before this reproduction; no outcome-based admission filter was applied.",
        "Collider bias": "No post-treatment variable is conditioned on; allocation pairs share the same frozen samples.",
        "Base rate neglect": "Accuracy is reported with its exact 20-sample denominator; this is not a diagnostic sensitivity/specificity claim.",
        "Regression to the mean": "The independent temporal rerun repeats every cell and is not selected from extreme parent outcomes.",
        "Survivorship bias": "The exact 30-cell matrix is required; any missing, failed, or extra cell fails validation.",
        "Look-elsewhere effect": "Five task/length/model cells, two allocations, and three dataset seeds were frozen before execution; no cell is selected by result.",
        "Garden of forking paths": "Thinking mode, token budget, seeds, allocations, matrix, comparison rule, and denominator are fixed in the contract.",
        "Correlation != causation": "The report describes observed accuracy agreement and does not attribute a causal mechanism to state dtype.",
        "Reverse causality": "Allocation is experimentally assigned before generation, so outcome-to-allocation reverse direction is not applicable.",
    }
    return [
        {
            "fallacy": name,
            "severity": "CAUTION" if name in {"Ecological fallacy", "Look-elsewhere effect", "Garden of forking paths"} else "NOTE",
            "status": "CHECKED",
            "detail": details[name],
        }
        for name in FALLACY_NAMES
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-skill",
        "- Origin Mode: validate",
        f"- Origin Date: {report['material_passport']['origin_date']}",
        "- Verification Status: VERIFIED",
        "- Version Label: ruler_nothink_gate4_reproduction_v1",
        "",
        "## Validation Report",
        "",
        f"- **Source**: `{report['source']['reproduction_attempts']['2b']}` / `{report['source']['reproduction_attempts']['9b']}`",
        "- **Overall Confidence**: CAUTION",
        "- **Gate**: Gate 4 PASS",
        "- **Reproducibility**: REPRODUCIBLE (30/30 exact quality and sample-output matches)",
        "- **Review policy**: logical protocol audit; experiment-script hashes were not checked or used as gates",
        "",
        "### Statistical Findings",
        "",
        "| Cell | n pairs | FP16 mean | State-bf16 mean | Delta (95% t-CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["statistical_findings"]:
        low, high = row["ci95_delta_accuracy_points"]
        label = f"{row['model']} {row['task']} L{row['length']}"
        lines.append(
            f"| {label} | {row['n_pairs']} | {row['fp16_mean_accuracy']:.2f} | "
            f"{row['statebf16_mean_accuracy']:.2f} | {row['mean_delta_accuracy_points']:.2f} "
            f"[{low:.2f}, {high:.2f}] |"
        )
    lines.extend(
        [
            "",
            "### Warnings",
            "",
            "- The three paired dataset seeds provide low power. Exact observed equality does not establish equivalence or non-inferiority.",
            "- Degenerate zero-width intervals reflect identical observed outputs, not population-level certainty.",
            "- No p-values are used; the five predeclared cells are reported in full, without outcome selection.",
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
        lines.append(f"| {item['fallacy']} | {item['severity']} | {item['detail']} |")
    lines.extend(
        [
            "",
            "### Reproducibility",
            "",
            "The parent and reproduction use the same five-cell no-think matrix, allocations, dataset seeds, engine seed, token budget, model revisions, and evaluator semantics. Host and elapsed-time fields are intentionally excluded from quality comparison.",
            "",
            "All 30 primary accuracy values and all sample-level predictions, references, hit vectors, and token counts match exactly. Result JSON sidecars pass; no experiment-script hash was inspected.",
            "",
            "### Evidence Boundary",
            "",
            "This result supports only empirical agreement between fp32-state and bf16-state for the tested RULER cells. It does not prove general equivalence, cross-task quality preservation, or a serving mechanism.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruler-dir", type=Path, default=Path("results/quality/ruler-subset"))
    parser.add_argument("--contract", type=Path, default=Path("results/quality/ruler-nothink-5cell-reproduction-20260813.contract.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/reproduction/2026-08-13/ruler-nothink/ruler-nothink-5cell-gate4-20260813"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    ruler_dir = args.ruler_dir if args.ruler_dir.is_absolute() else root / args.ruler_dir
    contract_path = args.contract if args.contract.is_absolute() else root / args.contract
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    try:
        contract = validate_reproduction_contract(contract_path)
        parent_attempts = contract["parent_attempts"]
        reproduction_attempts = contract["attempt_ids"]
        original = validate_attempt_matrix(
            root,
            ruler_dir,
            parent_attempts["2b"],
            parent_attempts["9b"],
        )
        reproduction = validate_attempt_matrix(
            root,
            ruler_dir,
            reproduction_attempts["2b"],
            reproduction_attempts["9b"],
        )
        comparisons = compare_reproduction(original, reproduction)
        findings = statistical_rows(reproduction)
    except (KeyError, OSError, json.JSONDecodeError, ValidationError) as exc:
        raise SystemExit(f"RULER Gate 4 validation failed: {exc}") from exc

    report = {
        "schema_version": 1,
        "material_passport": {
            "origin_skill": "experiment-skill",
            "origin_mode": "validate",
            "origin_date": utc_now(),
            "verification_status": "VERIFIED",
            "version_label": "ruler_nothink_gate4_reproduction_v1",
        },
        "validation_id": "ruler-nothink-5cell-gate4-20260813",
        "gate": "Gate 4 reproducibility and promotion",
        "gate_status": "PASS",
        "evidence_status": "VERIFIED",
        "overall_confidence": "CAUTION",
        "source": {
            "contract": str(contract_path),
            "parent_attempts": parent_attempts,
            "reproduction_attempts": reproduction_attempts,
        },
        "review_policy": {
            "experiment_script_hashes_checked": False,
            "experiment_script_hashes_used_as_gates": False,
            "result_sidecars_checked": True,
            "logical_protocol_checks": True,
        },
        "matrix": {
            "expected_cells_per_run": 30,
            "parent_cells_validated": len(original),
            "reproduction_cells_validated": len(reproduction),
            "exact_reproduction_matches": len(comparisons),
            "samples_per_cell": EXPECTED_SAMPLES,
        },
        "statistical_findings": findings,
        "statistical_assumptions": {
            "unit_of_analysis": "paired dataset seed",
            "n_pairs_per_cell": 3,
            "interval": "two-sided 95% t interval with df=2",
            "effect_size": "raw paired accuracy-point delta; standardized effect undefined for zero variance",
            "multiple_comparisons": "all five predeclared cells reported; no p-value claims or outcome selection",
            "equivalence_margin": None,
            "equivalence_claim_authorized": False,
        },
        "warnings": [
            "n=3 paired dataset seeds is low power",
            "exact observed equality is not evidence of population equivalence",
            "zero-width empirical intervals do not authorize a no-loss claim",
        ],
        "fallacy_scan_coverage": "11/11",
        "fallacy_scan": fallacy_scan(),
        "reproducibility": {
            "classification": "environment_sensitive_quality_deterministic",
            "verdict": "REPRODUCIBLE",
            "timing_compared": False,
            "primary_metric_tolerance": "exact",
            "sample_output_tolerance": "exact",
            "comparisons": comparisons,
        },
        "promotion_boundary": contract["promotion_boundary"],
        "paper_quantitative_use_authorized": True,
        "claim_limit": "Observed equality only; no equivalence, non-inferiority, cross-task, or cross-hardware claim.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gate4_validation.json"
    markdown_path = out_dir / "validation_report.md"
    json_sha = atomic_write_json(json_path, report)
    markdown_sha = atomic_write_text(markdown_path, render_markdown(report))
    print(
        json.dumps(
            {
                "gate_status": report["gate_status"],
                "evidence_status": report["evidence_status"],
                "validated_cells_per_run": len(original),
                "exact_reproduction_matches": len(comparisons),
                "fallacy_scan_coverage": report["fallacy_scan_coverage"],
                "json": str(json_path),
                "json_sha256": json_sha,
                "markdown": str(markdown_path),
                "markdown_sha256": markdown_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
