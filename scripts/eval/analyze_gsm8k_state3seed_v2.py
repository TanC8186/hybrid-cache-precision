"""Deprecated seed-iid GSM8K analysis entry point.

Dataset seeds sample overlapping GSM8K items, so treating seed means as iid
replicates understates the dependence in these frozen runs.  This command is
kept only to fail loudly for historical invocations.  Use
``scripts/eval/analyze_gsm8k_dependence.py`` instead.
"""

from __future__ import annotations

def main() -> int:
    raise SystemExit(
        "DEPRECATED: seed means are not iid because dataset-seed samples overlap. "
        "Use scripts/eval/analyze_gsm8k_dependence.py with the frozen nine-seed inputs."
    )


if __name__ == "__main__":
    raise SystemExit(main())
