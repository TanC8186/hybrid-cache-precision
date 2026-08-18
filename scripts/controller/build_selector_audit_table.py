"""Build a hash-verified selector decision audit in JSON and LaTeX."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REASON_LABELS = {
    "insufficient_allocator_equivalent_sequence_slots": "slots",
    "memory_budget_exceeded": "memory",
    "ttft_slo_violated": "TTFT",
    "tpot_slo_violated": "TPOT",
    "quality_guardrail_violated:gsm8k": "quality",
}


class AuditError(ValueError):
    """Raised when a frozen selector decision is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise AuditError(f"decision file is missing: {path}")
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise AuditError(f"decision SHA-256 sidecar is missing: {sidecar}")
    expected = sidecar.read_text(encoding="ascii").strip()
    if SHA256_RE.fullmatch(expected) is None or expected != digest:
        raise AuditError(f"decision SHA-256 verification failed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"decision root must be an object: {path}")
    return value, digest


def atomic_write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest


def require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{field} must be numeric")
    return float(value)


def normalize_decision(label: str, path: Path) -> dict[str, Any]:
    decision, digest = load_verified_json(path)
    status = decision.get("status")
    if status not in {"SELECTED", "NO_FEASIBLE_CANDIDATE"}:
        raise AuditError(f"{label}: unsupported decision status: {status!r}")
    request = decision.get("request")
    evaluations = decision.get("evaluations")
    if not isinstance(request, dict) or not isinstance(evaluations, list) or not evaluations:
        raise AuditError(f"{label}: request/evaluations are missing")
    quality_constraints = request.get("quality_constraints")
    memory_budget = request.get("memory_budget")
    slo = request.get("slo")
    if not all(isinstance(value, dict) for value in (quality_constraints, memory_budget, slo)):
        raise AuditError(f"{label}: request constraints are incomplete")
    minimum_quality = require_number(quality_constraints.get("gsm8k"), f"{label}.quality_floor")
    required_slots = require_number(
        request.get("required_allocator_equivalent_sequence_slots"),
        f"{label}.required_slots",
    )
    selected = decision.get("selected")
    selected_id = selected.get("config_id") if isinstance(selected, dict) else None
    rows: list[dict[str, Any]] = []
    feasible_ids: list[str] = []
    for index, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, dict):
            raise AuditError(f"{label}.evaluations[{index}] must be an object")
        config_id = evaluation.get("config_id")
        checks = evaluation.get("constraint_checks")
        if not isinstance(config_id, str) or not isinstance(checks, dict):
            raise AuditError(f"{label}.evaluations[{index}] is incomplete")
        quality = checks.get("quality", {}).get("gsm8k")
        serving = checks.get("serving")
        capacity = checks.get("capacity")
        if not all(isinstance(value, dict) for value in (quality, serving, capacity)):
            raise AuditError(f"{label}/{config_id}: constraint trace is incomplete")
        interval = quality.get("delta_ci95")
        reasons = evaluation.get("rejection_reasons")
        if not isinstance(interval, list) or len(interval) != 2 or not isinstance(reasons, list):
            raise AuditError(f"{label}/{config_id}: quality interval/reasons are malformed")
        feasible = evaluation.get("feasible") is True
        if feasible:
            feasible_ids.append(config_id)
        if feasible != (len(reasons) == 0):
            raise AuditError(f"{label}/{config_id}: feasibility disagrees with rejection reasons")
        rows.append(
            {
                "candidate": config_id,
                "feasible": feasible,
                "selected": config_id == selected_id,
                "objective_lcb_req_s": require_number(
                    evaluation.get("objective_lcb_req_s"), f"{label}/{config_id}.objective"
                ),
                "quality_ci95": [
                    require_number(interval[0], f"{label}/{config_id}.quality_ci95[0]"),
                    require_number(interval[1], f"{label}/{config_id}.quality_ci95[1]"),
                ],
                "allocator_equivalent_sequence_slots": require_number(
                    evaluation.get("allocator_equivalent_sequence_slots"),
                    f"{label}/{config_id}.allocator_slots",
                ),
                "p95_ttft_ucb_ms": require_number(
                    serving.get("p95_ttft_ucb_ms"), f"{label}/{config_id}.ttft_ucb"
                ),
                "p95_tpot_ucb_ms": require_number(
                    serving.get("p95_tpot_ucb_ms"), f"{label}/{config_id}.tpot_ucb"
                ),
                "rejection_reasons": list(reasons),
            }
        )
    if status == "SELECTED":
        if selected_id not in feasible_ids:
            raise AuditError(f"{label}: selected candidate is not feasible")
    elif selected is not None or feasible_ids:
        raise AuditError(f"{label}: no-feasible status disagrees with candidate trace")
    return {
        "label": label,
        "source": {"path": path.as_posix(), "sha256": digest},
        "status": status,
        "selected_config_id": selected_id,
        "selected_objective_lcb_req_s": (
            require_number(selected.get("objective_lcb_req_s"), f"{label}.selected_objective")
            if isinstance(selected, dict)
            else None
        ),
        "request": {
            "workload": request.get("workload"),
            "offered_rate_req_s": require_number(request.get("offered_rate_req_s"), f"{label}.rate"),
            "required_allocator_equivalent_sequence_slots": required_slots,
            "gsm8k_minimum_delta": minimum_quality,
            "p95_ttft_limit_ms": require_number(slo.get("p95_ttft_ms"), f"{label}.ttft_limit"),
            "p95_tpot_limit_ms": require_number(slo.get("p95_tpot_ms"), f"{label}.tpot_limit"),
            "max_cache_bytes": require_number(memory_budget.get("max_cache_bytes"), f"{label}.cache"),
        },
        "candidates": rows,
    }


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    return "".join(replacements.get(character, character) for character in value)


def format_reasons(row: dict[str, Any]) -> str:
    if row["selected"]:
        return r"\textbf{selected}"
    if row["feasible"]:
        return "feasible"
    return "reject: " + ", ".join(
        latex_escape(REASON_LABELS.get(reason, reason)) for reason in row["rejection_reasons"]
    )


def render_latex(decisions: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Frozen selector audit after dependence-aware GSM8K reanalysis. Quality values are percentage-point differences from full precision. Slots are allocator-equivalent sequence slots, not demonstrated scheduler concurrency.}",
        r"\label{tab:selector-audit}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Budget & Candidate & Required slots & Quality floor & Quality 95\% CI & Goodput LCB & Outcome \\",
        r" & & & (pp) & (pp) & (req/s) & \\",
        r"\midrule",
    ]
    for decision_index, decision in enumerate(decisions):
        request = decision["request"]
        for row_index, row in enumerate(decision["candidates"]):
            label = latex_escape(decision["label"]) if row_index == 0 else ""
            interval = row["quality_ci95"]
            lines.append(
                f"{label} & {latex_escape(row['candidate'])} & "
                f"{request['required_allocator_equivalent_sequence_slots']:.0f} & "
                f"{100.0 * request['gsm8k_minimum_delta']:+.1f} & "
                f"[{100.0 * interval[0]:+.2f}, {100.0 * interval[1]:+.2f}] & "
                f"{row['objective_lcb_req_s']:.3f} & {format_reasons(row)} \\\\"
            )
        if decision_index + 1 < len(decisions):
            lines.append(r"\addlinespace[2pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_audit(specifications: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    if not specifications:
        raise AuditError("at least one decision is required")
    labels = [label for label, _ in specifications]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise AuditError("decision labels must be unique and non-empty")
    decisions = [normalize_decision(label, path) for label, path in specifications]
    return {
        "schema_version": 1,
        "artifact": "joint_precision_selector_audit",
        "capacity_semantics": "allocator_equivalent_sequence_slots",
        "quality_interval_semantics": (
            "two-way CR1 cluster-robust 95% confidence interval over paired seed-item draws"
        ),
        "decisions": decisions,
    }


def parse_decision(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--decision must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("--decision must be LABEL=PATH")
    return label, Path(raw_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", action="append", type=parse_decision, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--tex-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.decision)
    json_payload = json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    json_digest = atomic_write(args.json_out, json_payload)
    tex_digest = atomic_write(args.tex_out, render_latex(audit["decisions"]).encode("ascii"))
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "json_sha256": json_digest,
                "tex_out": str(args.tex_out),
                "tex_sha256": tex_digest,
                "n_decisions": len(audit["decisions"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
