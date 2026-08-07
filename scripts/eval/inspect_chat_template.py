"""Print the Qwen3.5-2B chat template snippet and enable_thinking usage."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    path = Path(
        "/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master/tokenizer_config.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    template = data.get("chat_template", "")
    print("template len:", len(template))
    print(template[:2200])
    idx = template.find("enable_thinking")
    if idx >= 0:
        print("\n--- around enable_thinking ---")
        print(template[max(0, idx - 500) : idx + 700])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
