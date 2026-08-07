"""Inspect one RULER-subset cell: summary fields + first case."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    rec = json.loads(path.read_text(encoding="utf-8"))
    print(
        {
            key: rec.get(key)
            for key in (
                "task",
                "task_type",
                "length",
                "allocation",
                "seed",
                "num_samples",
                "accuracy",
                "tokens_to_generate",
                "elapsed_s",
                "status",
            )
        }
    )
    print("data_sha256:", (rec.get("data_sha256") or "")[:16])
    first = rec["cases"][0]
    print("first prediction:", repr(first["prediction"][:160]))
    print("first references:", first["references"])
    print("first hits:", first["hits"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
