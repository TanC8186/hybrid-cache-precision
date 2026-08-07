"""Verify RULER-subset datasets: row counts, sha256 sidecars, answer_prefix present."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/ruler")
    ok = True
    for data_file in sorted(root.glob("*/validation.jsonl")):
        rows = [
            json.loads(line)
            for line in data_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        missing_prefix = [row.get("index") for row in rows if "answer_prefix" not in row]
        sidecar = Path(str(data_file) + ".sha256")
        digest = sha256_file(data_file)
        sidecar_ok = sidecar.exists() and sidecar.read_text(encoding="ascii").strip() == digest
        row_ok = len(rows) == 20 and not missing_prefix
        ok = ok and row_ok and sidecar_ok
        print(
            f"{data_file} rows={len(rows)} missing_prefix={len(missing_prefix)} "
            f"sha={digest[:12]} sidecar={sidecar_ok}"
        )
        if row_ok:
            sample = rows[0]
            print(
                "  sample:",
                {
                    "index": sample["index"],
                    "length": sample.get("length"),
                    "outputs": sample["outputs"],
                    "answer_prefix": sample.get("answer_prefix", "")[:60],
                },
            )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
