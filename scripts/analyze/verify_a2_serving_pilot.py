"""Audit the A2 packed serving MVEx and failed pilot without rewriting evidence."""

from __future__ import annotations

import argparse
import json
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
    write_text_with_hash,
)

MVEX_ID = "a2-packed-serving-mvex-d1d52c4-westd-01"
PILOT_ID = "a2-packed-serving-pilot-d1d52c4-westd-01"
ROOT_COMMIT = "d1d52c49de57ab61776dfeb7cabd16ff8b6bd40c"
VLLM_COMMIT = "55f47685a553ad8d776c464c59785399a98c7185"
FATAL_SIGNATURES = (
    "EngineCore encountered a fatal error",
    "CUDA error: an illegal instruction was encountered",
)


def request_accounting(result: Mapping[str, Any], expected: int) -> dict[str, Any]:
    completed = int(result["completed"])
    failed = int(result["failed"])
    require(completed + failed == expected, "completed + failed does not match expected requests")

    detail_lengths = {
        name: len(result[name])
        for name in ("ttfts", "itls", "output_lens", "start_times", "errors")
    }
    require(
        all(length == expected for length in detail_lengths.values()),
        f"detailed result length mismatch: {detail_lengths}",
    )
    errors = list(result["errors"])
    detailed_failures = sum(bool(error) for error in errors)
    require(detailed_failures == failed, "reported failures do not match non-empty errors")
    require(expected - detailed_failures == completed, "reported completions do not match empty errors")
    return {
        "expected": expected,
        "completed": completed,
        "failed": failed,
        "request_conservation": True,
        "detail_lengths": detail_lengths,
        "detailed_failures": detailed_failures,
    }


def detect_fatal_signatures(log_text: str) -> dict[str, bool]:
    return {signature: signature in log_text for signature in FATAL_SIGNATURES}


def classify_pilot(
    *,
    supervisor_exit_code: int,
    summary_counts: Mapping[str, int],
    fatal_signatures: Mapping[str, bool],
    failed_sample_zero_success: bool,
) -> str:
    if (
        supervisor_exit_code != 0
        and int(summary_counts.get("failed", 0)) > 0
        and all(fatal_signatures.values())
        and failed_sample_zero_success
    ):
        return "PILOT_FAILED_RUNTIME_CUDA_ILLEGAL_INSTRUCTION"
    return "PILOT_INTEGRITY_REVIEW_REQUIRED"


def audit_common_attempt(attempt_dir: Path, attempt_id: str) -> dict[str, Any]:
    for name in ("attempt_contract.json", "environment.json", "summary.json"):
        verify_sidecar(attempt_dir / f"{name}.sha256")

    contract = load_json(attempt_dir / "attempt_contract.json")
    environment = load_json(attempt_dir / "environment.json")
    summary = load_json(attempt_dir / "summary.json")
    require(contract["attempt_id"] == attempt_id, f"{attempt_id}: attempt ID mismatch")
    require(contract["git_commit"] == ROOT_COMMIT, f"{attempt_id}: root commit mismatch")
    require(contract["vllm_source_commit"] == VLLM_COMMIT, f"{attempt_id}: vLLM commit mismatch")
    require(
        environment["root_git"]["commit"] == ROOT_COMMIT,
        f"{attempt_id}: environment root commit mismatch",
    )
    require(
        environment["vllm_source_commit"] == VLLM_COMMIT,
        f"{attempt_id}: environment vLLM commit mismatch",
    )
    return {
        "contract": contract,
        "environment": environment,
        "summary": summary,
    }


def sample_plan(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["sample_id"]): item for item in contract["plan"]}


def audit_completed_sample(
    attempt_dir: Path,
    plan: Mapping[str, Mapping[str, Any]],
    sample_id: str,
) -> dict[str, Any]:
    sample_dir = attempt_dir / "samples" / sample_id
    for name in ("contract.json", "result.json", "analysis.json"):
        verify_sidecar(sample_dir / f"{name}.sha256")

    status = load_json(sample_dir / "status.json")
    result = load_json(sample_dir / "result.json")
    analysis = load_json(sample_dir / "analysis.json")
    require(status["status"] == "completed_validated", f"{sample_id}: status mismatch")
    require(status["result_sha256"] == sha256_file(sample_dir / "result.json"), f"{sample_id}: result hash")
    require(
        status["analysis_sha256"] == sha256_file(sample_dir / "analysis.json"),
        f"{sample_id}: analysis hash",
    )
    accounting = request_accounting(result, int(plan[sample_id]["num_prompts"]))
    require(analysis["completed"] == accounting["completed"], f"{sample_id}: analysis completed mismatch")
    require(analysis["failed"] == accounting["failed"], f"{sample_id}: analysis failed mismatch")
    return {
        "sample_id": sample_id,
        "status": status["status"],
        "accounting": accounting,
        "arrival_span_over_target": analysis["arrival_span_over_target"],
        "reported_ttft_p99_ms": analysis["reported_ttft_p99_ms"],
        "reported_tpot_p99_ms": analysis["reported_tpot_p99_ms"],
        "sustainable_thresholds": [
            threshold
            for threshold, item in analysis["slo_sweep"].items()
            if item["sustainable"]
        ],
    }


def audit_failed_sample(
    attempt_dir: Path,
    plan: Mapping[str, Mapping[str, Any]],
    sample_id: str,
) -> dict[str, Any]:
    sample_dir = attempt_dir / "samples" / sample_id
    verify_sidecar(sample_dir / "contract.json.sha256")
    status = load_json(sample_dir / "status.json")
    require(status["status"] == "failed", f"{sample_id}: expected failed status")
    raw_result = sample_dir / "work" / "result.json"
    require(raw_result.is_file(), f"{sample_id}: failed raw result missing")
    result = load_json(raw_result)
    accounting = request_accounting(result, int(plan[sample_id]["num_prompts"]))
    return {
        "sample_id": sample_id,
        "status": status["status"],
        "failure_stage": status.get("failure_stage"),
        "error": status.get("error"),
        "returncode": status.get("returncode"),
        "accounting": accounting,
        "raw_result_path": str(raw_result),
        "raw_result_sha256": sha256_file(raw_result),
    }


def server_log(attempt_dir: Path) -> Path:
    matches = list((attempt_dir / "servers").glob("*/*/server.log"))
    require(len(matches) == 1, f"expected one server log in {attempt_dir}, found {len(matches)}")
    return matches[0]


def supervisor_exit(supervisor_dir: Path) -> int:
    require((supervisor_dir / "finished_at.txt").is_file(), "supervisor finish timestamp missing")
    return int((supervisor_dir / "exit_code.txt").read_text(encoding="ascii").strip())


def build_validation_markdown(report: Mapping[str, Any]) -> str:
    mvex = report["mvex"]
    pilot = report["pilot"]
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_packed_serving_pilot_audit_v1

## Validation Report

- **MVEx**: `{MVEX_ID}`
- **Pilot**: `{PILOT_ID}`
- **Pilot Verdict**: `{pilot['verdict']}`
- **Evidence Status**: `QUARANTINED`
- **Overall A2 Status**: `{report['a2_overall_status']}`

### MVEx

The MVEx completed {mvex['sample']['accounting']['completed']:,}/
{mvex['sample']['accounting']['expected']:,} requests with zero failures,
arrival ratio `{mvex['sample']['arrival_span_over_target']:.6f}`, P99 TTFT
`{mvex['sample']['reported_ttft_p99_ms']:.2f}` ms, and P99 TPOT
`{mvex['sample']['reported_tpot_p99_ms']:.2f}` ms. It remains `UNVERIFIED`.

### Failed Pilot

| Sample | Completed | Failed | Status |
|---|---:|---:|---|
| packed/random/r30/s7 | {pilot['rate30']['accounting']['completed']:,} | {pilot['rate30']['accounting']['failed']:,} | completed_validated, then quarantined with the session |
| packed/random/r40/s7 | {pilot['rate40']['accounting']['completed']:,} | {pilot['rate40']['accounting']['failed']:,} | failed result validation |
| packed/random/r50/s7 | 0 | 0 | not started |

All issued requests are accounted for; there are no silent exclusions. However,
the shared server session logged both `EngineCore encountered a fatal error` and
`CUDA error: an illegal instruction was encountered`. The rate-40 result had no
successful request, the supervisor exited nonzero, and rate 50 was not started.
The pilot therefore fails Gate 2 and must not be resumed or promoted.

### Next Gate

Run a new parent-linked diagnostic attempt with `CUDA_LAUNCH_BLOCKING=1`. Do not
start packed comparative or formal serving experiments until the failing kernel
is identified and a fresh MVEx plus pilot pass.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    attempts_dir = raw_dir / "attempts"
    supervisors_dir = raw_dir / "supervisors"

    mvex_dir = attempts_dir / MVEX_ID
    pilot_dir = attempts_dir / PILOT_ID
    mvex_common = audit_common_attempt(mvex_dir, MVEX_ID)
    pilot_common = audit_common_attempt(pilot_dir, PILOT_ID)
    mvex_plan = sample_plan(mvex_common["contract"])
    pilot_plan = sample_plan(pilot_common["contract"])

    mvex_sample_id = "packed_per_layer__random__r30__s7"
    mvex_sample = audit_completed_sample(mvex_dir, mvex_plan, mvex_sample_id)
    mvex_log_text = server_log(mvex_dir).read_text(encoding="utf-8", errors="replace")
    mvex_fatals = detect_fatal_signatures(mvex_log_text)
    require(not any(mvex_fatals.values()), "MVEx server log contains a fatal signature")
    require(supervisor_exit(supervisors_dir / MVEX_ID) == 0, "MVEx supervisor did not exit zero")
    require(mvex_common["summary"]["counts"] == {"completed_validated": 1}, "MVEx summary drift")
    require(mvex_sample["accounting"]["failed"] == 0, "MVEx has failed requests")

    rate30_id = "packed_per_layer__random__r30__s7"
    rate40_id = "packed_per_layer__random__r40__s7"
    rate30 = audit_completed_sample(pilot_dir, pilot_plan, rate30_id)
    rate40 = audit_failed_sample(pilot_dir, pilot_plan, rate40_id)
    pilot_log_path = server_log(pilot_dir)
    pilot_fatals = detect_fatal_signatures(
        pilot_log_path.read_text(encoding="utf-8", errors="replace")
    )
    pilot_exit = supervisor_exit(supervisors_dir / PILOT_ID)
    summary_counts = pilot_common["summary"]["counts"]
    require(
        summary_counts == {"completed_validated": 1, "failed": 1, "not_started": 1},
        "pilot summary drift",
    )
    verdict = classify_pilot(
        supervisor_exit_code=pilot_exit,
        summary_counts=summary_counts,
        fatal_signatures=pilot_fatals,
        failed_sample_zero_success=rate40["accounting"]["completed"] == 0,
    )
    require(
        verdict == "PILOT_FAILED_RUNTIME_CUDA_ILLEGAL_INSTRUCTION",
        "pilot failure classification did not close",
    )

    report = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "verification_status": "ANALYZED",
        "evidence_status": "QUARANTINED",
        "a2_overall_status": "PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING",
        "mvex": {
            "attempt_id": MVEX_ID,
            "supervisor_exit_code": 0,
            "fatal_signatures": mvex_fatals,
            "sample": mvex_sample,
        },
        "pilot": {
            "attempt_id": PILOT_ID,
            "supervisor_exit_code": pilot_exit,
            "summary_counts": summary_counts,
            "fatal_signatures": pilot_fatals,
            "server_log_path": str(pilot_log_path),
            "server_log_sha256": sha256_file(pilot_log_path),
            "rate30": rate30,
            "rate40": rate40,
            "rate50_status": "not_started",
            "request_conservation": (
                rate30["accounting"]["request_conservation"]
                and rate40["accounting"]["request_conservation"]
            ),
            "silent_exclusions": 0,
            "pilot_gate_passed": False,
            "verdict": verdict,
        },
        "interpretation": (
            "The packed serving MVEx passed once, but the subsequent pilot hit a CUDA illegal "
            "instruction during the rate-30 sample. All requests were accounted for, yet the "
            "server-session fatal error quarantines the full pilot and blocks expansion."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_sha = write_json_with_hash(output_dir / "pilot_audit_report.json", report)
    validation_sha = write_text_with_hash(
        output_dir / "validation_report.md",
        build_validation_markdown(report),
    )
    source_files = sorted(path for path in raw_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "files": [
            {
                "path": str(path.relative_to(raw_dir)).replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
        "generated": {
            "pilot_audit_report.json": report_sha,
            "validation_report.md": validation_sha,
        },
    }
    write_json_with_hash(output_dir / "artifact_sha256_manifest.json", manifest)
    print(
        json.dumps(
            {
                "mvex": f"{mvex_sample['accounting']['completed']}/"
                f"{mvex_sample['accounting']['expected']}",
                "pilot_rate30": f"{rate30['accounting']['completed']}/"
                f"{rate30['accounting']['expected']}",
                "pilot_rate40": f"{rate40['accounting']['completed']}/"
                f"{rate40['accounting']['expected']}",
                "silent_exclusions": 0,
                "verdict": verdict,
                "a2_overall_status": report["a2_overall_status"],
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
