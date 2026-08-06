"""Package an immutable boundary of the A2 ShareGPT formal attempt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ATTEMPT_ID = (
    "a2-comparative-serving-sharegpt300-formal-v3-piecewise-3108650-westd-01"
)
CONFIG_NAME = "a2_comparative_piecewise_protocol_v3_sharegpt300_formal.yaml"
DEPLOYMENT_FILES = (
    "a2-comparative-formal-v3-3108650.bundle",
    "a2-comparative-formal-v3-3108650.bundle.sha256",
)
PLAN_SIZE = 63
SLICE_SIZE = 5


class PackagingError(RuntimeError):
    """Raised when the requested evidence boundary is not packageable."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PackagingError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_sidecar(path: Path) -> str:
    return path.read_text(encoding="ascii").strip().split()[0]


def copy_file(source: Path, destination: Path) -> None:
    require(source.is_file(), f"missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    require(source.is_dir(), f"missing source directory: {source}")
    require(not destination.exists(), f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def verify_snapshot(snapshot_dir: Path) -> tuple[dict[str, Any], str]:
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    sidecar_path = snapshot_dir / "snapshot_manifest.json.sha256"
    require(manifest_path.is_file(), f"snapshot manifest missing: {manifest_path}")
    require(sidecar_path.is_file(), f"snapshot sidecar missing: {sidecar_path}")
    manifest_sha256 = sha256_file(manifest_path)
    require(
        manifest_sha256 == read_sidecar(sidecar_path),
        "snapshot manifest sidecar mismatch",
    )
    manifest = load_json(manifest_path)
    for item in manifest["files"]:
        path = snapshot_dir / item["path"]
        require(path.is_file(), f"snapshot file missing: {path}")
        require(path.stat().st_size == item["size"], f"snapshot size mismatch: {path}")
        require(sha256_file(path) == item["sha256"], f"snapshot hash mismatch: {path}")
    return manifest, manifest_sha256


def select_invocations(
    attempt_dir: Path,
    captured_at: str,
    expected_count: int,
) -> list[Path]:
    selected: list[tuple[str, str, Path]] = []
    for path in (attempt_dir / "invocations").glob("*.json"):
        invoked_at = str(load_json(path)["invoked_at"])
        if invoked_at <= captured_at:
            selected.append((invoked_at, path.name, path))
    selected.sort()
    require(
        len(selected) == expected_count,
        f"invocation boundary count: expected {expected_count}, got {len(selected)}",
    )
    return [item[2] for item in selected]


def select_servers(
    attempt_dir: Path,
    captured_at: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for status_path in (attempt_dir / "servers").glob("*/*/status.json"):
        status = load_json(status_path)
        contract = load_json(status_path.with_name("contract.json"))
        started_at = str(contract["started_at"])
        if started_at > captured_at:
            continue
        require(status["status"] == "stopped", f"server not stopped: {status_path}")
        require(status["returncode"] == 0, f"server return code: {status_path}")
        require(status["exception"] is None, f"server exception: {status_path}")
        selected.append(
            {
                "allocation": status_path.parents[1].name,
                "session": status_path.parent.name,
                "started_at": started_at,
                "status": status["status"],
                "source": status_path.parent,
            }
        )
    selected.sort(key=lambda item: (item["started_at"], item["allocation"], item["session"]))
    require(
        len(selected) == expected_count,
        f"server boundary count: expected {expected_count}, got {len(selected)}",
    )
    return selected


def write_json_with_sidecar(path: Path, value: Any) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def package_slice(
    evidence_root: Path,
    source_repo: Path,
    slice_number: int,
    suffix: str,
) -> dict[str, Any]:
    require(1 <= slice_number <= 13, "slice number outside ShareGPT formal plan")
    slice_id = f"slice-{slice_number:03d}"
    completed_count = min(slice_number * SLICE_SIZE, PLAN_SIZE)
    output_stem = (
        "a2-protocol-v3-sharegpt300-formal-3108650-westd-01-"
        f"{slice_id}{suffix}"
    )
    staging_dir = evidence_root / "staging" / output_stem
    archive_path = evidence_root / "archives" / f"{output_stem}.tar.gz"
    archive_sidecar = archive_path.with_name(archive_path.name + ".sha256")
    temp_staging = staging_dir.with_name(f".{staging_dir.name}.tmp-{os.getpid()}")
    temp_archive = archive_path.with_name(f".{archive_path.name}.tmp-{os.getpid()}")

    for path in (staging_dir, archive_path, archive_sidecar, temp_staging, temp_archive):
        require(not path.exists(), f"refusing to overwrite existing path: {path}")

    attempt_dir = evidence_root / "attempts" / ATTEMPT_ID
    snapshot_dir = (
        evidence_root / "boundary-snapshots" / ATTEMPT_ID / slice_id
    )
    supervisor_root = evidence_root / "supervisors" / ATTEMPT_ID
    snapshot, snapshot_sha256 = verify_snapshot(snapshot_dir)
    require(snapshot["attempt_id"] == ATTEMPT_ID, "snapshot attempt ID")
    require(snapshot["slice_id"] == slice_id, "snapshot slice ID")
    require(
        snapshot["expected_completed_samples"] == completed_count,
        "snapshot completed-sample boundary",
    )

    summary = load_json(snapshot_dir / "attempt" / "summary.json")
    contract = load_json(snapshot_dir / "attempt" / "attempt_contract.json")
    completed_sample_ids = [
        str(item["sample_id"])
        for item in summary["samples"]
        if item["status"] == "completed_validated"
    ]
    plan_sample_ids = [
        str(item["sample_id"]) for item in contract["plan"][:completed_count]
    ]
    require(
        summary["counts"].get("completed_validated") == completed_count,
        "summary completed count",
    )
    require(
        completed_sample_ids == plan_sample_ids,
        "completed samples do not match the frozen plan prefix",
    )

    postcheck = load_json(
        snapshot_dir / "supervisor" / "postcheck.json"
    )
    require(postcheck["status"] == "PASSED", "slice postcheck status")
    require(postcheck["completed_samples"] == completed_count, "postcheck sample count")
    require(postcheck["failed_requests"] == 0, "postcheck failed requests")
    invocations = select_invocations(attempt_dir, snapshot["captured_at"], slice_number)
    servers = select_servers(
        attempt_dir,
        snapshot["captured_at"],
        int(postcheck["server_sessions"]),
    )

    temp_staging.mkdir(parents=True)
    try:
        for name in (
            "attempt_contract.json",
            "attempt_contract.json.sha256",
            "environment.json",
            "environment.json.sha256",
            "summary.json",
            "summary.json.sha256",
        ):
            copy_file(
                snapshot_dir / "attempt" / name,
                temp_staging / "attempts" / ATTEMPT_ID / name,
            )

        for invocation in invocations:
            copy_file(
                invocation,
                temp_staging / "attempts" / ATTEMPT_ID / "invocations" / invocation.name,
            )

        for sample_id in completed_sample_ids:
            sample_source = attempt_dir / "samples" / sample_id
            sample_status = load_json(sample_source / "status.json")
            require(
                sample_status["status"] == "completed_validated",
                f"sample is not completed_validated: {sample_id}",
            )
            copy_tree(
                sample_source,
                temp_staging / "attempts" / ATTEMPT_ID / "samples" / sample_id,
            )

        server_manifest: list[dict[str, Any]] = []
        for server in servers:
            copy_tree(
                server["source"],
                temp_staging
                / "attempts"
                / ATTEMPT_ID
                / "servers"
                / server["allocation"]
                / server["session"],
            )
            server_manifest.append(
                {key: value for key, value in server.items() if key != "source"}
            )

        supervisors: list[str] = []
        for index in range(1, slice_number + 1):
            supervisor_id = f"slice-{index:03d}"
            copy_tree(
                supervisor_root / supervisor_id,
                temp_staging / "supervisors" / ATTEMPT_ID / supervisor_id,
            )
            supervisors.append(supervisor_id)

        for name in (
            f"{ATTEMPT_ID}.json",
            f"{ATTEMPT_ID}.dry-run.json",
        ):
            copy_file(
                evidence_root / "contracts" / name,
                temp_staging / "contracts" / name,
            )
        for name in DEPLOYMENT_FILES:
            copy_file(
                evidence_root / "deployment" / name,
                temp_staging / "deployment" / name,
            )
        copy_file(
            source_repo / "experiments" / "configs" / CONFIG_NAME,
            temp_staging / "experiments" / "configs" / CONFIG_NAME,
        )

        selection_manifest = {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "attempt_id": ATTEMPT_ID,
            "slice_id": slice_id,
            "completed_samples": completed_count,
            "sample_ids": completed_sample_ids,
            "invocations": [path.name for path in invocations],
            "server_sessions": server_manifest,
            "supervisors": supervisors,
            "snapshot_captured_at": snapshot["captured_at"],
            "snapshot_manifest_sha256": snapshot_sha256,
            "source_repo": str(source_repo),
        }
        write_json_with_sidecar(
            temp_staging / "archive_selection_manifest.json",
            selection_manifest,
        )
        os.replace(temp_staging, staging_dir)

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "-czf", str(temp_archive), "-C", str(staging_dir), "."],
            check=True,
        )
        subprocess.run(["gzip", "-t", str(temp_archive)], check=True)
        archive_sha256 = sha256_file(temp_archive)
        archive_size = temp_archive.stat().st_size
        os.replace(temp_archive, archive_path)
        archive_sidecar.write_text(
            f"{archive_sha256}  {archive_path.name}\n",
            encoding="ascii",
        )
        return {
            "archive": str(archive_path),
            "sha256": archive_sha256,
            "size": archive_size,
            "staging": str(staging_dir),
            "completed_samples": completed_count,
            "server_sessions": len(servers),
            "invocations": len(invocations),
        }
    except Exception:
        # Keep partial paths for forensic inspection instead of hiding a failed build.
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--slice-number", type=int, required=True)
    parser.add_argument(
        "--suffix",
        default="-corrected",
        help="Suffix appended to the immutable staging and archive name.",
    )
    parser.add_argument(
        "--wait-for-boundary",
        action="store_true",
        help="Wait until the immutable boundary snapshot exists before packaging.",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=int,
        default=7200,
        help="Maximum boundary wait time when --wait-for-boundary is set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.wait_for_boundary:
        snapshot_sidecar = (
            args.evidence_root.resolve()
            / "boundary-snapshots"
            / ATTEMPT_ID
            / f"slice-{args.slice_number:03d}"
            / "snapshot_manifest.json.sha256"
        )
        failure_marker = (
            args.evidence_root.resolve()
            / "orchestrators"
            / "sharegpt-formal-slices-006-013-v2"
            / "failure.txt"
        )
        deadline = time.monotonic() + args.wait_timeout_seconds
        while not snapshot_sidecar.is_file():
            require(
                not failure_marker.is_file(),
                f"formal orchestrator failed before boundary: {failure_marker}",
            )
            require(
                time.monotonic() < deadline,
                f"timed out waiting for boundary: {snapshot_sidecar}",
            )
            time.sleep(10)
    result = package_slice(
        args.evidence_root.resolve(),
        args.source_repo.resolve(),
        args.slice_number,
        args.suffix,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PackagingError as exc:
        print(f"packaging failed: {exc}")
        raise SystemExit(2) from exc
