"""Run the frozen deployment-epoch joint precision selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from kvcache.policy import NoFeasibleCandidate, PolicyInputError, select_joint_precision


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write(path: Path, value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)
    digest = sha256_file(path)
    atomic_write_bytes(path.with_suffix(path.suffix + ".sha256"), f"{digest}\n".encode("ascii"))
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    request = json.loads(args.request.read_text(encoding="utf-8"))
    exit_code = 0
    started_ns = time.perf_counter_ns()
    try:
        report = select_joint_precision(profile, request)
    except NoFeasibleCandidate as error:
        report = error.report
        exit_code = 2
    except PolicyInputError as error:
        report = {
            "schema_version": 2,
            "policy": "joint_precision_deployment_epoch_v2",
            "status": "INVALID_INPUT",
            "error": {"type": type(error).__name__, "message": str(error)},
            "selected": None,
        }
        exit_code = 3
    report["decision_latency_ms"] = (time.perf_counter_ns() - started_ns) / 1_000_000
    report["material_passport"] = {
        "origin_skill": "experiment-skill",
        "origin_mode": "run",
        "origin_date": datetime.now(timezone.utc).date().isoformat(),
        "verification_status": "UNVERIFIED",
        "version_label": "joint_precision_decision_v1",
    }
    report["inputs"] = {
        "profile": str(args.profile),
        "profile_sha256": sha256_file(args.profile),
        "request": str(args.request),
        "request_sha256": sha256_file(args.request),
    }
    digest = atomic_write(args.out, report)
    print(json.dumps({"status": report["status"], "out": str(args.out), "sha256": digest}, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
