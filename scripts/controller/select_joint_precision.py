"""Run the frozen deployment-epoch joint precision selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from kvcache.policy import NoFeasibleCandidate, select_joint_precision


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}\n", encoding="ascii")
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
    try:
        report = select_joint_precision(profile, request)
    except NoFeasibleCandidate as error:
        report = error.report
        exit_code = 2
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
