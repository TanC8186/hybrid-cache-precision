"""Dependence-aware paired analysis for seed-subsampled GSM8K runs.

The same GSM8K item can be sampled by more than one dataset seed, so seed
means are neither independent replications nor the only dependence source.
This analysis keeps every paired seed-item draw, clusters the intercept-only
contrast by both item and dataset seed, and uses a conservative t reference
with the smaller cluster degrees of freedom.  An equal-item-weight cluster
bootstrap is reported as a sensitivity estimand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


DEFAULT_ALLOCATIONS = (
    "fp16",
    "fp16_statebf16",
    "uniform_int4",
    "uniform_int4_statebf16",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest


def _round(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def load_cells(
    attempt_dir: Path,
    allocations: Iterable[str],
    seeds: Iterable[int],
) -> dict[tuple[str, int], dict[str, Any]]:
    required = {(allocation, seed) for allocation in allocations for seed in seeds}
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(attempt_dir.glob("gsm8k__*.json")):
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.is_file() or sha256_file(path) != sidecar.read_text(encoding="ascii").strip():
            raise ValueError(f"SHA-256 verification failed: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record.get("allocation"), record.get("seed"))
        if key not in required:
            continue
        if key in cells:
            raise ValueError(f"duplicate allocation/seed cell: {key}")
        if record.get("status") != "completed_validated" or record.get("bench") != "gsm8k":
            raise ValueError(f"invalid completed GSM8K cell: {path}")
        indices = record.get("sampled_indices")
        cases = record.get("cases")
        if not isinstance(indices, list) or not isinstance(cases, list) or len(indices) != len(cases):
            raise ValueError(f"sampled_indices/cases mismatch: {path}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"within-seed sampling must be without replacement: {path}")
        if record.get("num_samples") != len(cases) or record.get("seed_semantics") is None:
            raise ValueError(f"incomplete sampling metadata: {path}")
        hits = [_require_bool(case.get("hit"), f"{path}: cases[{index}].hit") for index, case in enumerate(cases)]
        observed_accuracy = sum(hits) / len(hits)
        if not math.isclose(float(record.get("accuracy", math.nan)), observed_accuracy, abs_tol=1e-12):
            raise ValueError(f"accuracy does not match case.hit values: {path}")
        cells[key] = record
    missing = sorted(required - set(cells))
    if missing:
        raise ValueError(f"incomplete allocation/seed cells: {missing}")
    return cells


def validate_pairing(
    cells: dict[tuple[str, int], dict[str, Any]],
    allocations: list[str],
    seeds: list[int],
) -> None:
    baseline = allocations[0]
    for seed in seeds:
        expected_indices = cells[(baseline, seed)]["sampled_indices"]
        expected_questions = [case.get("question") for case in cells[(baseline, seed)]["cases"]]
        for allocation in allocations[1:]:
            record = cells[(allocation, seed)]
            if record["sampled_indices"] != expected_indices:
                raise ValueError(f"sampled item order differs for {allocation}, seed={seed}")
            questions = [case.get("question") for case in record["cases"]]
            if questions != expected_questions:
                raise ValueError(f"question order differs for {allocation}, seed={seed}")


def paired_draws(
    cells: dict[tuple[str, int], dict[str, Any]],
    left: str,
    right: str,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    differences: list[float] = []
    item_groups: list[int] = []
    seed_groups: list[int] = []
    for seed in seeds:
        left_record = cells[(left, seed)]
        right_record = cells[(right, seed)]
        for item, left_case, right_case in zip(
            left_record["sampled_indices"],
            left_record["cases"],
            right_record["cases"],
        ):
            differences.append(float(right_case["hit"]) - float(left_case["hit"]))
            item_groups.append(int(item))
            seed_groups.append(seed)
    return (
        np.asarray(differences, dtype=float),
        np.asarray(item_groups, dtype=int),
        np.asarray(seed_groups, dtype=int),
    )


def two_way_cluster_inference(
    differences: np.ndarray,
    item_groups: np.ndarray,
    seed_groups: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    if not (len(differences) == len(item_groups) == len(seed_groups)) or len(differences) < 2:
        raise ValueError("clustered contrast arrays must have equal non-trivial length")
    item_count = len(np.unique(item_groups))
    seed_count = len(np.unique(seed_groups))
    if item_count < 2 or seed_count < 2:
        raise ValueError("two-way clustering requires at least two item and seed clusters")
    fitted = sm.OLS(differences, np.ones((len(differences), 1), dtype=float)).fit()
    covariance, _, _ = cov_cluster_2groups(
        fitted,
        item_groups,
        seed_groups,
        use_correction=True,
    )
    variance = float(covariance[0, 0])
    if variance < -1e-12:
        raise ValueError(f"two-way cluster variance is negative: {variance}")
    standard_error = math.sqrt(max(variance, 0.0))
    estimate = float(fitted.params[0])
    degrees_of_freedom = min(item_count - 1, seed_count - 1)
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom))
    if standard_error == 0.0:
        p_value = 1.0 if estimate == 0.0 else 0.0
    else:
        p_value = float(2.0 * stats.t.sf(abs(estimate / standard_error), degrees_of_freedom))
    interval = [estimate - critical * standard_error, estimate + critical * standard_error]
    return {
        "method": "intercept_only_ols_two_way_cluster_robust_cr1",
        "estimate": _round(estimate),
        "standard_error": _round(standard_error),
        "reference_distribution": "student_t",
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": _round(p_value),
        "ci_level": 1.0 - alpha,
        "ci": [_round(value) for value in interval],
        "n_seed_item_draws": len(differences),
        "n_item_clusters": item_count,
        "n_seed_clusters": seed_count,
    }


def item_equal_weight_sensitivity(
    differences: np.ndarray,
    item_groups: np.ndarray,
    *,
    alpha: float,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for item, difference in zip(item_groups.tolist(), differences.tolist()):
        grouped[item].append(difference)
    item_means = np.asarray([np.mean(grouped[item]) for item in sorted(grouped)], dtype=float)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(bootstrap_reps, dtype=float)
    chunk_size = 512
    for start in range(0, bootstrap_reps, chunk_size):
        stop = min(start + chunk_size, bootstrap_reps)
        sampled = rng.integers(0, len(item_means), size=(stop - start, len(item_means)))
        bootstrap[start:stop] = item_means[sampled].mean(axis=1)
    interval = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimand": "equal_weight_mean_of_unique_item_level_paired_differences",
        "estimate": _round(float(item_means.mean())),
        "method": "percentile_cluster_bootstrap_over_unique_items",
        "ci_level": 1.0 - alpha,
        "ci": [_round(value) for value in interval],
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_seed": bootstrap_seed,
        "n_unique_items": len(item_means),
        "gain_items": int(np.sum(item_means > 0)),
        "loss_items": int(np.sum(item_means < 0)),
        "tie_items": int(np.sum(item_means == 0)),
    }


def allocation_diagnostics(
    cells: dict[tuple[str, int], dict[str, Any]],
    allocation: str,
    seeds: list[int],
) -> dict[str, Any]:
    outcomes: dict[int, list[bool]] = defaultdict(list)
    for seed in seeds:
        record = cells[(allocation, seed)]
        for item, case in zip(record["sampled_indices"], record["cases"]):
            outcomes[int(item)].append(bool(case["hit"]))
    inconsistent = sum(1 for values in outcomes.values() if len(values) > 1 and len(set(values)) > 1)
    return {
        "n_unique_items": len(outcomes),
        "repeated_items_with_inconsistent_outcomes": inconsistent,
    }


def analyze_attempt(
    attempt_dir: Path,
    allocations: list[str],
    seeds: list[int],
    *,
    alpha: float = 0.05,
    bootstrap_reps: int = 20_000,
    bootstrap_seed: int = 20_260_814,
) -> dict[str, Any]:
    if len(allocations) < 2 or allocations[0] != "fp16":
        raise ValueError("allocations must start with the fp16 baseline and include a contrast")
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("at least two unique dataset seeds are required")
    if bootstrap_reps < 1000:
        raise ValueError("bootstrap_reps must be at least 1000")
    cells = load_cells(attempt_dir, allocations, seeds)
    validate_pairing(cells, allocations, seeds)

    item_sets = {
        seed: set(int(item) for item in cells[(allocations[0], seed)]["sampled_indices"])
        for seed in seeds
    }
    pairwise_overlaps = [
        len(item_sets[left] & item_sets[right])
        for index, left in enumerate(seeds)
        for right in seeds[index + 1 :]
    ]
    unique_items = set().union(*item_sets.values())
    per_allocation_diagnostics = {
        allocation: allocation_diagnostics(cells, allocation, seeds)
        for allocation in allocations
    }

    rows: list[dict[str, Any]] = []
    for allocation in allocations:
        per_seed = {str(seed): float(cells[(allocation, seed)]["accuracy"]) for seed in seeds}
        scores = np.asarray(list(per_seed.values()), dtype=float)
        row: dict[str, Any] = {
            "allocation": allocation,
            "n_items_per_seed": len(cells[(allocation, seeds[0])]["sampled_indices"]),
            "n_dataset_seeds": len(seeds),
            "n_unique_items": per_allocation_diagnostics[allocation]["n_unique_items"],
            "per_seed": {key: _round(value) for key, value in per_seed.items()},
            "mean_accuracy_over_seed_item_draws": _round(float(scores.mean())),
            "sd_of_seed_accuracies_descriptive_only": _round(float(scores.std(ddof=1))),
        }
        if allocation != allocations[0]:
            differences, items, seed_groups = paired_draws(cells, allocations[0], allocation, seeds)
            primary = two_way_cluster_inference(differences, items, seed_groups, alpha)
            sensitivity = item_equal_weight_sensitivity(
                differences,
                items,
                alpha=alpha,
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=bootstrap_seed,
            )
            row.update(
                {
                    "delta_vs_fp16": primary["estimate"],
                    "ci95_vs_fp16": primary["ci"],
                    "cluster_robust_inference": primary,
                    "item_equal_weight_sensitivity": sensitivity,
                }
            )
        rows.append(row)

    stacking_marginal = None
    if "uniform_int4" in allocations and "uniform_int4_statebf16" in allocations:
        differences, items, seed_groups = paired_draws(
            cells,
            "uniform_int4",
            "uniform_int4_statebf16",
            seeds,
        )
        primary = two_way_cluster_inference(differences, items, seed_groups, alpha)
        stacking_marginal = {
            "compare": "uniform_int4_statebf16 vs uniform_int4",
            "mean": primary["estimate"],
            "ci95": primary["ci"],
            "cluster_robust_inference": primary,
            "item_equal_weight_sensitivity": item_equal_weight_sensitivity(
                differences,
                items,
                alpha=alpha,
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=bootstrap_seed,
            ),
        }

    return {
        "schema_version": 2,
        "material_passport": {
            "origin_mode": "cpu_reanalysis",
            "verification_status": "ANALYZED",
            "version_label": "gsm8k_dependence_aware_v1",
        },
        "attempt": attempt_dir.name,
        "bench": "gsm8k",
        "accuracy_field": "cases[].hit",
        "protocol": (
            "no-think, greedy, max_tokens=1024; 200 items sampled without replacement per dataset seed; "
            "the same seed uses the same ordered subset in every allocation"
        ),
        "primary_estimand": (
            "mean paired accuracy difference over the observed seed-item draws generated by the frozen "
            "dataset-seed sampling protocol"
        ),
        "inference_note": (
            "Items recur across dataset seeds, so seed-level means are not treated as independent iid "
            "replications. The primary interval clusters by item and dataset seed; with nine seed clusters, "
            "the t reference has 8 degrees of freedom. The item bootstrap targets a distinct equal-item-weight estimand."
        ),
        "diagnostics": {
            "n_seed_item_draws": sum(len(values) for values in item_sets.values()),
            "n_unique_items": len(unique_items),
            "pairwise_seed_overlap_min": min(pairwise_overlaps),
            "pairwise_seed_overlap_median": _round(float(np.median(pairwise_overlaps))),
            "pairwise_seed_overlap_max": max(pairwise_overlaps),
            "allocation_outcome_stability": per_allocation_diagnostics,
        },
        "rows": rows,
        "stacking_marginal": stacking_marginal,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("results/quality/reasoning"))
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allocations", default=",".join(DEFAULT_ALLOCATIONS))
    parser.add_argument("--seeds", default="7,42,2026,11,23,31,47,73,97")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_814)
    args = parser.parse_args()
    allocations = [value.strip() for value in args.allocations.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    result = analyze_attempt(
        args.dir / args.attempt,
        allocations,
        seeds,
        alpha=args.alpha,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    digest = atomic_write_json(args.out, result)
    print(json.dumps({"out": str(args.out), "sha256": digest, "diagnostics": result["diagnostics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
