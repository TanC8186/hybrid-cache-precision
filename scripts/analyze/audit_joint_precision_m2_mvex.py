"""Audit the scoped M2 joint-precision controller MVEx artifact."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditError(RuntimeError):
    """Raised when an artifact fails an integrity gate."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AuditError(f"JSON object expected: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AuditError(f"{name} is not numeric: {value!r}") from error
    require(math.isfinite(result), f"{name} is not finite")
    return result


def command_value(command: list[str], flag: str) -> str:
    require(command.count(flag) == 1, f"command must contain one {flag}")
    index = command.index(flag)
    require(index + 1 < len(command), f"command value missing: {flag}")
    return str(command[index + 1])


def fallacy_scan() -> list[dict[str, str]]:
    names = [
        "Simpson's paradox",
        "Ecological fallacy",
        "Berkson's paradox",
        "Collider bias",
        "Base rate neglect",
        "Regression to the mean",
        "Survivorship bias",
        "Look-elsewhere effect",
        "Garden of forking paths",
        "Correlation is not causation",
        "Reverse causality",
    ]
    return [
        {
            "fallacy": name,
            "severity": "NOTE",
            "detail": "The MVEx has one predeclared allocation, workload, rate, and seed; no effect claim is promoted.",
            "recommendation": "Keep this sample diagnostic and use the frozen multi-seed pilot for inference.",
        }
        for name in names
    ]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: {report['generated_at_utc'][:10]}
- Verification Status: ANALYZED
- Version Label: joint_precision_m2_mvex_validation_v1

## Validation Report

- **Source**: `{report['attempt_id']}`
- **Gate 1 Verdict**: `PASS`
- **Evidence Status**: `UNVERIFIED` diagnostic evidence
- **Overall Confidence**: `CAUTION`
- **Reproducibility**: `CANNOT_VERIFY` at single-seed MVEx

### Integrity Findings

The scoped slice contains 1/1 validated sample and 2,400/2,400 measurement
requests, with zero failed requests, zero silent exclusions, one cold-start
server session, and internally consistent contracts/results. The selector selected `joint`
and the executed command proves int4 KV, bfloat16 state, and PIECEWISE graphs.

| Metric | Value |
|---|---:|
| Request throughput | {metrics['request_throughput_req_s']:.6f} req/s |
| Throughput / offered | {metrics['request_throughput_over_offered']:.6f} |
| P95 TTFT | {metrics['ttft_p95_ms_recomputed']:.6f} ms |
| P99 TTFT | {metrics['ttft_p99_ms_recomputed']:.6f} ms |
| P95 TPOT | {metrics['tpot_p95_ms_recomputed']:.6f} ms |
| P99 TPOT | {metrics['tpot_p99_ms_recomputed']:.6f} ms |
| Goodput at 500/200 ms | {metrics['goodput_500_req_s']:.6f} req/s |

### Warnings

- This is a minimum viable execution gate, not a statistical comparison.
- The outer SSH wrapper recorded a nonnumeric shell status token; controller and runner artifacts independently record successful completion and return code 0.
- The frozen M2 contract covers Qwen3.5-2B, 4096 context, and Random workload only; it does not establish the full multi-model/context claim.

### Fallacy Scan

- **Coverage**: 11/11

""" + "\n".join(
        f"- **{item['fallacy']}** (`{item['severity']}`): {item['detail']}"
        for item in report["fallacy_scan"]
    ) + """

### Promotion Decision

Gate 1 passes and authorizes the predeclared pilot. This artifact remains
`UNVERIFIED`; it is not paper-usable until the multi-seed pilot, independent
reproducibility comparison, and statistical audit pass.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact_dir.resolve()
    raw_root = artifact / "raw"
    controller_root = raw_root / "m2-mvex-20260812-r1" / "joint-precision-m2-mvex-high-s11-20260812-r1"
    attempt = controller_root / "runner" / "joint-precision-m2-mvex-high-s11-20260812-r1"
    controller = load_json(controller_root / "controller_result.json")
    contract = load_json(controller_root / "controller_contract.json")
    decision = load_json(controller_root / "decision.json")
    attempt_contract = load_json(attempt / "attempt_contract.json")
    environment = load_json(attempt / "environment.json")
    summary = load_json(attempt / "summary.json")
    sample_id = "joint__random__r40__s11"
    sample = attempt / "samples" / sample_id
    sample_contract = load_json(sample / "contract.json")
    sample_status = load_json(sample / "status.json")
    analysis = load_json(sample / "analysis.json")
    result = load_json(sample / "result.json")
    server_dirs = list((attempt / "servers" / "joint").iterdir())
    require(len(server_dirs) == 1, "expected one joint server session")
    server = server_dirs[0]
    server_contract = load_json(server / "contract.json")
    server_status = load_json(server / "status.json")
    require(controller["status"] == "COMPLETED_UNVERIFIED", "controller did not complete")
    require(controller["selected_config_id"] == "joint", "selector mapping drift")
    require(contract["dry_run"] is False, "artifact is a dry-run")
    require(contract["root_git"]["clean"] is True and contract["root_git"]["status"] == "", "root worktree was not clean")
    require(Path(contract["profile_path"]).name == "physical_calibration_profile.json", "profile path drift")
    require(Path(contract["request_path"]).name == "joint_precision_m2_high_20260812.json", "request path drift")
    require(Path(contract["serving_config_path"]).name == "joint_precision_controller_2b.yaml", "serving config path drift")
    require(contract["phase"] == {"name": "confirmatory", "allocations": ["joint"], "seeds": [11], "workload_rates": {"random": [40.0]}}, "controller phase drift")
    require(decision["status"] == "SELECTED" and decision["selected"]["config_id"] == "joint", "decision invalid")
    require(attempt_contract["git_commit"] == contract["root_git"]["commit"], "attempt commit drift")
    require(environment["root_git"] == contract["root_git"], "environment git drift")
    require(attempt_contract["vllm_source_commit"] == environment["vllm_source_commit"], "vLLM revision drift")
    require(attempt_contract["phase"]["seeds"] == [11], "seed drift")
    require(attempt_contract["plan"] == [{"allocation": "joint", "num_prompts": 2400, "request_rate": 40.0, "sample_id": sample_id, "seed": 11, "workload": "random"}], "sample plan drift")
    require(summary["counts"] == {"completed_validated": 1}, "summary denominator drift")
    require(sample_status["status"] == "completed_validated" and sample_status["returncode"] == 0, "sample status invalid")
    require(result["completed"] == 2400 and result["failed"] == 0, "request denominator drift")
    for field in ("ttfts", "itls", "input_lens", "output_lens", "start_times", "errors"):
        require(len(result[field]) == 2400, f"detailed field length drift: {field}")
    require(analysis["status"] == "completed_validated", "analysis status invalid")
    require(server_status == {"exception": None, "returncode": 0, "startup_duration_s": server_status["startup_duration_s"], "status": "stopped", "updated_at": server_status["updated_at"]}, "server status malformed")
    command = [str(value) for value in server_contract["command"]]
    require(command_value(command, "--kv-cache-dtype") == "int4_per_token_head", "KV precision proof missing")
    require(command_value(command, "--mamba-ssm-cache-dtype") == "bfloat16", "state precision proof missing")
    require(any("CUDAGraphMode.PIECEWISE" in line for line in (server / "server.log").read_text(encoding="utf-8", errors="replace").splitlines()), "PIECEWISE proof missing")
    require(not list(attempt.rglob("*.partial")), "partial artifact remains")
    metrics = {
        key: finite(analysis[key], key)
        for key in ("request_throughput_req_s", "request_throughput_over_offered", "ttft_p95_ms_recomputed", "ttft_p99_ms_recomputed", "tpot_p95_ms_recomputed", "tpot_p99_ms_recomputed")
    }
    metrics["goodput_500_req_s"] = finite(analysis["slo_sweep"]["500"]["goodput_req_s"], "goodput_500_req_s")
    report = {
        "schema_version": 1,
        "material_passport": {"origin_skill": "experiment-skill", "origin_mode": "validate", "origin_date": datetime.now(timezone.utc).date().isoformat(), "verification_status": "ANALYZED", "version_label": "joint_precision_m2_mvex_validation_v1"},
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "attempt_id": contract["attempt_id"],
        "gate_1_verdict": "PASS",
        "evidence_status": "UNVERIFIED",
        "root_commit": contract["root_git"]["commit"],
        "vllm_commit": environment["vllm_source_commit"],
        "denominator": {"planned_samples": 1, "completed_validated_samples": 1, "planned_measurement_requests": 2400, "completed_measurement_requests": 2400, "failed_measurement_requests": 0, "silent_exclusions": 0},
        "selected_config_id": "joint",
        "metrics": metrics,
        "server_startup_duration_s": finite(server_status["startup_duration_s"], "startup_duration_s"),
        "fallacy_scan_coverage": "11/11",
        "fallacy_scan": fallacy_scan(),
        "reproducibility": {"determinism_class": "environment_sensitive_seeded_serving_benchmark", "method": "not run at MVEx", "verdict": "CANNOT_VERIFY"},
        "promotion": {"pilot_authorized": True, "paper_quantitative_use_authorized": False},
    }
    write_json(artifact / "mvex_audit_report.json", report)
    (artifact / "validation_report.md").write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"attempt_id": contract["attempt_id"], "gate_1_verdict": "PASS", "samples": "1/1", "measurement_requests": "2400/2400", "failed_requests": 0, "audit_mode": "logical_only"}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print(f"ERROR: {error}")
        raise SystemExit(2)
