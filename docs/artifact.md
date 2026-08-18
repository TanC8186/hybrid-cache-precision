# Artifact guide

This guide separates integrity checking, result regeneration, and full GPU
re-execution. The distinction is deliberate: the first two tiers use committed
evidence, while the third requires licensed external assets and the original
hardware class.

## 1. Supported environment

- Linux or WSL2
- Python 3.10-3.12 (the maintained environment uses Python 3.12)
- CPU-only verification: no model or dataset download
- Full experiments: one NVIDIA RTX 5090 32 GB, CUDA-compatible PyTorch, and a
  vLLM build for `sm_120`

Create the analysis environment with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,dev]"
```

## 2. Tier A: integrity and unit tests

```bash
./reproduce.sh verify
```

This runs:

```bash
python -m pytest -q
python paper/mlsys2026/figures/verify_figure_data.py
```

The figure verifier checks the checksums and values used by the manuscript,
including the corrected 2026-08-14 capacity analysis and dependence-aware
GSM8K intervals. A passing integrity check does not relabel exploratory or
failed-gate evidence as verified.

Hashes for tracked textual evidence use Git-canonical LF bytes. The verifier
normalizes a stale Windows CRLF checkout before hashing so the contract remains
identical across native Windows, WSL, and Linux clones.

## 3. Tier B: regenerate paper outputs

Regenerate all vector figures and their ignored raster previews:

```bash
./reproduce.sh figures
```

Check the generated capacity table without rewriting it:

```bash
python scripts/bench/build_capacity_matrix_table.py --check
```

Build the IEEE/DLS paper when `latexmk` is installed:

```bash
./reproduce.sh paper
```

The MLSys and DLS directories share the same evidence but use different venue
templates. Do not count them as independent studies or replications.

## 4. Tier C: reconstruct the GPU environment

The vLLM working tree is intentionally ignored because compiled trees are
large and architecture-specific. Reconstruct the exact two-commit patch stack
from the tracked mailbox patch:

```bash
git clone --filter=blob:none https://github.com/vllm-project/vllm vendor/vllm
git -C vendor/vllm checkout e2fa28594f7baad142a426b0b6a2cfe2c79201c7
git -C vendor/vllm am ../vllm-patches/per-layer-kv-a2.patch
git -C vendor/vllm rev-parse HEAD
```

The expected final commit is
`55f47685a553ad8d776c464c59785399a98c7185`. The patch stack adds the
per-layer page-group implementation used by the archived experiments. Build
it only inside the target CUDA environment:

```bash
bash scripts/build_vllm.sh
./env_check.sh
```

Model weights and datasets are not redistributed. Obtain Qwen3.5-2B/9B and
the evaluation corpora under their original terms, then verify dataset
revisions and hashes against `data/MANIFEST.yaml`. ShareGPT traces have
separate redistribution constraints.

Frozen experiment YAML files preserve the absolute paths used on the original
server as provenance. Copy a config and adapt its `environment` paths for a
new machine; do not edit a hash-addressed archived contract in place.

The project contains dedicated launchers rather than one universal GPU
command. The main entry points are:

| Study | Configuration or launcher |
|---|---|
| Joint precision controller | `configs/experiments/joint_precision_controller_2b.yaml` and `scripts/controller/` |
| Capacity phase diagram | `scripts/bench/run_capacity_phase_diagram.sh` |
| GSM8K dependence-aware analysis | `scripts/eval/analyze_gsm8k_dependence.py` |
| Mechanism isolation | `configs/experiments/m3_mechanism_isolation_2b.yaml` and `scripts/analyze/analyze_m3_mechanism_isolation.py` |
| Serving stability audit | `scripts/quality/analyze_m4_formal.py` and `scripts/quality/verify_m4_gate4.py` |

Every formal run must start from a clean Git tree and archive its resolved
configuration, code commit, environment probe, seeds, and raw-output hashes.

## 5. Canonical evidence

| Claim family | Source of record |
|---|---|
| Capacity and residuals | `results/verified/2026-08-14/capacity-*-corrected*` |
| GSM8K clustered inference | `results/quality/*dependence-aware-20260814.json` |
| Selector decisions | `results/verified/2026-08-14/controller-decisions/` |
| Controller profiles | `results/verified/2026-08-14/controller-profile/` |
| Four-configuration stability | `results/reproduction/2026-08-13/m4-four-config/analysis-r3/` |
| Manuscript value ledger | `paper/mlsys2026/figures/verify_figure_data.py` |

Older `screen`, `pilot`, and superseded analyses are retained for auditability
but are not interchangeable with the canonical files above.

## 6. Repository and release policy

Large raw server archives remain local under ignored paths. New Git commits
should contain compact aggregate outputs, contracts, checksums, analyzers, and
the minimum raw evidence needed to audit a claim. Never commit credentials,
model weights, private traces, virtual environments, or compiled vLLM trees.

Before making this repository public:

1. Confirm double-blind policy and remove internal reviewer-process files.
2. Replace manuscript author placeholders and add `CITATION.cff`.
3. Choose a public source-code license and audit every vendored file.
4. Recheck dataset and trace redistribution rights.
5. Run a full history secret scan and consider a curated release history.
6. Create a tagged artifact release and archive it with a persistent DOI.
