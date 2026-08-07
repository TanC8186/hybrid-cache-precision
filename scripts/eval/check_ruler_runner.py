"""Preflight check for the RULER-subset runner (no GPU needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ruler_quality  # noqa: PLC0415

    print("tasks_base:", sorted(ruler_quality.load_tasks_base().keys()))
    print("metrics:", sorted(ruler_quality.load_metrics().keys()))

    data_file = Path("data/ruler/ruler_niah_single_L4096/validation.jsonl")
    rows = [
        json.loads(line)
        for line in data_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample = rows[0]
    prompt = sample["input"] + sample.get("answer_prefix", "")
    print("rows:", len(rows))
    print("prompt_tail:", prompt[-100:].replace("\n", "\\n"))
    print("outputs:", sample["outputs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
