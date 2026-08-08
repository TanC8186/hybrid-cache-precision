"""Complete the cached Qwen2.5-7B-Instruct snapshot (pure-attention control)."""

from __future__ import annotations

from modelscope import snapshot_download


def main() -> int:
    target = "/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"
    path = snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir=target)
    print("DONE", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
