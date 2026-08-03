"""One-shot precise insertion of kv_cache_dtype_per_layer into CacheConfig
construction in vllm/engine/arg_utils.py (server site-packages)."""
import sys

path = "/root/autodl-tmp/MLSys_Research/.venv/lib/python3.12/site-packages/vllm/engine/arg_utils.py"

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

anchor = "            kv_cache_dtype_skip_layers=self.kv_cache_dtype_skip_layers,\n"
insert = "            kv_cache_dtype_per_layer=self.kv_cache_dtype_per_layer,\n"

# Find the anchor line
idxs = [i for i, l in enumerate(lines) if l == anchor]
if not idxs:
    print("ERROR: anchor line not found. Aborting (no changes).")
    sys.exit(1)
if len(idxs) > 1:
    print(f"ERROR: anchor line not unique: {len(idxs)} occurrences. Aborting.")
    sys.exit(1)

i = idxs[0]
# Guard: don't double-insert if already present right after
if lines[i + 1] == insert:
    print("ALREADY PATCHED: insertion already present. No changes.")
    sys.exit(0)

lines.insert(i + 1, insert)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("PATCHED OK")
print("--- region ---")
for j in range(i, i + 3):
    print(f"{j+1}: {lines[j]}", end="")
print("--- end ---")
