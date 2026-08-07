"""Analyze R5 TurboQuant/FP8 protocol-v3 serving attempts.

For each attempt (MVEx/Pilot/Formal): verify request conservation from the
runner's atomic artifacts and produce per-(allocation, workload, TTFT
threshold) sustainable boundaries across all seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sidecar(path: Path) -> None:
    digest = sha256_file(path)
    recorded = path.read_text(encoding="ascii").strip()
    if digest != recorded:
        raise SystemExit(f"sidecar mismatch: {path}")


def audit_attempt(raw_dir: Path, attempt_id: str) -> dict:
    attempt_dir = raw_dir / "attempts" / attempt_id
    if not attempt_dir.is_dir():
        raise SystemExit(f"missing attempt: {attempt_dir}")
    contract = load_json(attempt_dir / "attempt_contract.json")
    summary = load_json(attempt_dir / "summary.json")
    plan = {str(item["sample_id"]): item for item in contract["plan"]}
    samples = []
    for sample_dir in sorted((attempt_dir / "samples").glob("*")):
        if not sample_dir.is_dir():
            continue
        for name in ("contract.json", "result.json", "analysis.json", "status.json"):
            verify_sidecar(sample_dir / f"{name}.sha256")
        sample_contract = load_json(sample_dir / "contract.json")
        result = load_json(sample_dir / "result.json")
        analysis = load_json(sample_dir / "analysis.json")
        status = load_json(sample_dir / "status.json")
        if status["status"] != "completed_validated":
            continue
        expected = int(plan[sample_dir.name]["num_prompts"])
        if int(analysis["completed"]) + int(analysis["failed"]) != expected:
            raise SystemExit(
                f"{attempt_id}/{sample_dir.name}: denominator mismatch "
                f"(completed+failed != plan num_prompts {expected})"
            )
        samples.append(
            {
                "sample_id": sample_dir.name,
                "allocation": sample_contract["allocation"],
                "workload": sample_contract["workload"],
                "offered_rate": float(sample_contract["request_rate"]),
                "seed": int(sample_contract["seed"]),
                "completed": int(analysis["completed"]),
                "failed": int(analysis["failed"]),
                "expected": expected,
                "request_throughput_over_offered": float(analysis["request_throughput_over_offered"]),
                "ttft_p99_ms": float(analysis["reported_ttft_p99_ms"]),
                "tpot_p99_ms": float(analysis["reported_tpot_p99_ms"]),
                "goodput_over_offered": {
                    int(threshold): float(item["goodput_over_offered"])
                    for threshold, item in analysis["slo_sweep"].items()
                },
            }
        )
    expected_samples = len(contract["plan"])
    if len(samples) != expected_samples:
        raise SystemExit(
            f"{attempt_id}: completed {len(samples)}/{expected_samples} samples "
            f"(summary={summary['counts']})"
        )
    return {
        "attempt_id": attempt_id,
        "phase": contract["phase"]["name"],
        "git_commit": contract["git_commit"],
        "vllm_source_commit": contract.get("vllm_source_commit"),
        "config_sha256": contract["config_sha256"],
        "samples": samples,
    }


def boundaries(samples: list[dict]) -> dict:
    keys = sorted({(s["allocation"], s["workload"]) for s in samples})
    out = {}
    for allocation, workload in keys:
        cell_samples = [s for s in samples if s["allocation"] == allocation and s["workload"] == workload]
        seeds = {s["seed"] for s in cell_samples}
        rates = sorted({s["offered_rate"] for s in cell_samples})
        thresholds = sorted({t for s in cell_samples for t in s["goodput_over_offered"]})
        row = {}
        for threshold in thresholds:
            sustainable_rates = []
            for rate in rates:
                rate_samples = [s for s in cell_samples if s["offered_rate"] == rate]
                if len(rate_samples) != len(seeds):
                    raise SystemExit(f"missing seed sample: {allocation}/{workload} r={rate}")
                if all(s["goodput_over_offered"][threshold] >= 0.95 for s in rate_samples):
                    sustainable_rates.append(rate)
            row[str(threshold)] = max(sustainable_rates) if sustainable_rates else None
        out[f"{allocation}__{workload}"] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--attempt-ids", required=True)
    ap.add_argument("--out", default="results/quality/r5-serving-analysis.json")
    args = ap.parse_args()

    attempts = [
        audit_attempt(args.raw_dir.resolve(), attempt_id.strip())
        for attempt_id in args.attempt_ids.split(",")
        if attempt_id.strip()
    ]
    all_samples = [sample for attempt in attempts for sample in attempt["samples"]]
    result = {
        "schema_version": 1,
        "attempts": [
            {
                "attempt_id": a["attempt_id"],
                "phase": a["phase"],
                "git_commit": a["git_commit"],
                "vllm_source_commit": a["vllm_source_commit"],
                "config_sha256": a["config_sha256"],
                "num_samples": len(a["samples"]),
            }
            for a in attempts
        ],
        "boundaries": boundaries(all_samples),
        "samples": all_samples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["boundaries"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
