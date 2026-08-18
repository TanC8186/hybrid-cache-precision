#!/usr/bin/env bash
# Reconstruct the ignored vLLM working tree from upstream plus tracked patches.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VLLM="$ROOT/vendor/vllm"
BASE_SHA="e2fa28594f7baad142a426b0b6a2cfe2c79201c7"
EXPECTED_SHA="55f47685a553ad8d776c464c59785399a98c7185"
PATCH="$ROOT/vendor/vllm-patches/per-layer-kv-a2.patch"

if git -C "$VLLM" rev-parse --git-dir >/dev/null 2>&1; then
  actual="$(git -C "$VLLM" rev-parse HEAD)"
  if [ "$actual" != "$EXPECTED_SHA" ]; then
    echo "ERROR: vendor/vllm is at $actual; expected $EXPECTED_SHA." >&2
    exit 1
  fi
  if [ -n "$(git -C "$VLLM" status --porcelain)" ]; then
    echo "ERROR: vendor/vllm has local changes; use a clean working tree." >&2
    git -C "$VLLM" status --short >&2
    exit 1
  fi
  echo "vLLM is already pinned at $EXPECTED_SHA"
  exit 0
fi

if [ -e "$VLLM" ]; then
  echo "ERROR: $VLLM exists but is not a Git working tree." >&2
  exit 1
fi

git clone --filter=blob:none https://github.com/vllm-project/vllm "$VLLM"
git -C "$VLLM" checkout "$BASE_SHA"
git -C "$VLLM" am "$PATCH"

actual="$(git -C "$VLLM" rev-parse HEAD)"
if [ "$actual" != "$EXPECTED_SHA" ]; then
  echo "ERROR: reconstructed vLLM is $actual; expected $EXPECTED_SHA." >&2
  exit 1
fi

echo "vLLM reconstructed at $EXPECTED_SHA"
