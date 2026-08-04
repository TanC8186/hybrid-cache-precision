"""Transactionally deploy committed vLLM source files over an installed wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

RUNTIME_FILES = (
    "vllm/config/cache.py",
    "vllm/engine/arg_utils.py",
    "vllm/model_executor/layers/attention/attention.py",
    "vllm/utils/torch_utils.py",
    "vllm/v1/core/kv_cache_utils.py",
    "vllm/v1/kv_cache_interface.py",
    "vllm/v1/worker/gpu_model_runner.py",
)


class DeployError(RuntimeError):
    """Raised when deployment cannot be completed atomically."""


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    digest = sha256_file(path)
    hash_path = path.with_suffix(path.suffix + ".sha256")
    hash_path.write_text(f"{digest}\n", encoding="ascii")
    return digest


def run_capture(command: Sequence[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise DeployError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout[-4000:]}")
    return completed.stdout.strip()


def detect_target_root(python: Path) -> Path:
    output = run_capture(
        [
            str(python),
            "-c",
            ("import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent.parent)"),
        ]
    )
    return Path(output).resolve()


def absolute_preserving_symlinks(path: Path) -> Path:
    """Make a path absolute without resolving a venv interpreter symlink."""
    return Path(os.path.abspath(path.expanduser()))


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    shutil.copy2(source, tmp)
    os.replace(tmp, target)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = args.source_root.resolve()
    python = absolute_preserving_symlinks(args.python)
    target_root = args.target_root.resolve() if args.target_root else detect_target_root(python)
    manifest_dir = args.manifest_dir.resolve()
    source_commit = run_capture(["git", "rev-parse", "HEAD"], cwd=source_root)
    source_status = run_capture(["git", "status", "--porcelain=v1"], cwd=source_root)
    if source_status:
        raise DeployError("source vLLM checkout is dirty:\n" + source_status)
    if args.expected_source_commit and source_commit != args.expected_source_commit:
        raise DeployError(f"source commit mismatch: expected={args.expected_source_commit}, actual={source_commit}")

    records: list[dict[str, Any]] = []
    for relative in RUNTIME_FILES:
        source = source_root / relative
        target = target_root / relative
        if not source.is_file():
            raise DeployError(f"source file missing: {source}")
        if not target.is_file():
            raise DeployError(f"installed target file missing: {target}")
        records.append(
            {
                "relative_path": relative,
                "source": str(source),
                "target": str(target),
                "source_sha256": sha256_file(source),
                "before_sha256": sha256_file(target),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "started_at": utc_timestamp(),
        "source_root": str(source_root),
        "source_commit": source_commit,
        "python": str(python),
        "target_root": str(target_root),
        "dry_run": args.dry_run,
        "files": records,
    }
    if args.dry_run:
        manifest["status"] = "dry_run"
        digest = atomic_write_json(manifest_dir / "deployment.json", manifest)
        print(json.dumps({"status": "dry_run", "manifest_sha256": digest}, indent=2))
        return 0

    backup_root = manifest_dir / "backup"
    backup_root.mkdir(parents=True, exist_ok=False)
    deployed: list[dict[str, Any]] = []
    try:
        for record in records:
            relative = record["relative_path"]
            target = Path(record["target"])
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            record["backup"] = str(backup)

        for record in records:
            source = Path(record["source"])
            target = Path(record["target"])
            atomic_copy(source, target)
            record["after_sha256"] = sha256_file(target)
            if record["after_sha256"] != record["source_sha256"]:
                raise DeployError(f"post-copy hash mismatch: {target}")
            deployed.append(record)

        run_capture(
            [
                str(python),
                "-m",
                "py_compile",
                *(record["target"] for record in records),
            ]
        )
        package_info = run_capture(
            [
                str(python),
                "-c",
                ("import json, vllm; print(json.dumps({'version': vllm.__version__, 'file': vllm.__file__}))"),
            ]
        )
        manifest["package_info"] = json.loads(package_info)
        manifest["status"] = "deployed"
        manifest["finished_at"] = utc_timestamp()
    except Exception as exc:
        for record in records:
            backup_value = record.get("backup")
            if backup_value:
                atomic_copy(Path(backup_value), Path(record["target"]))
        manifest["status"] = "rolled_back"
        manifest["error"] = str(exc)
        manifest["finished_at"] = utc_timestamp()
        atomic_write_json(manifest_dir / "deployment.json", manifest)
        raise

    digest = atomic_write_json(manifest_dir / "deployment.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "source_commit": source_commit,
                "target_root": str(target_root),
                "manifest_sha256": digest,
                "files_deployed": len(deployed),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DeployError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
