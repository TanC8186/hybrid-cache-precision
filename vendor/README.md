# Pinned upstream components

`vendor/vllm` is an ignored working tree, not a Git submodule. This avoids
committing architecture-specific binaries while keeping the source state
reconstructable from the tracked mailbox patch.

## Reconstruct vLLM

```bash
git clone --filter=blob:none https://github.com/vllm-project/vllm vendor/vllm
git -C vendor/vllm checkout e2fa28594f7baad142a426b0b6a2cfe2c79201c7
git -C vendor/vllm am ../vllm-patches/per-layer-kv-a2.patch
git -C vendor/vllm rev-parse HEAD
```

Expected final commit:
`55f47685a553ad8d776c464c59785399a98c7185`.

The two commits add packed per-layer KV page groups and balance the Mamba cache
groups. `per-layer-kv-dtype.diff` is retained as an earlier implementation
record; `per-layer-kv-a2.patch` is the reconstructable patch stack used by the
archived experiments.

Build vLLM only in Linux/WSL or the target CUDA container. Ada (`sm_89`) and
Blackwell (`sm_120`) builds are not binary-compatible.

`vendor/ruler` contains the licensed subset of RULER utilities required by the
evaluation harness and retains its upstream license.
