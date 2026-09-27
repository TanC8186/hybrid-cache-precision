# Precision Accounting for Hybrid LLM Serving

Research artifact for **Precision Accounting for KV and Recurrent-State
Caches in Hybrid LLM Serving**, published at DLS 2026 (the 2nd Workshop on
Data Security and LLM Safety in Smart Systems, co-located with IEEE MASS
2026). The project studies how attention KV precision and recurrent-state
precision share one vLLM cache budget in hybrid linear-attention models.

[Camera-ready paper (DLS 2026)](paper/dls2026/main.pdf) |
[Artifact guide](docs/artifact.md) |
[Environment](docs/environment.md) |
[Canonical results](results/README.md)

![Measured capacity and the continuous accounting model](paper/mlsys2026/figures/vector_redesign/fig1_capacity.svg)

## Citation and license

If you use this artifact, cite the DLS 2026 paper; machine-readable metadata
is in [`CITATION.cff`](CITATION.cff). The camera-ready references this
repository under the `dls2026-camera-ready` tag.

The repository's own code and documentation are released under the
[MIT license](LICENSE). Vendored components and third-party data retain their
original terms; see [`vendor/README.md`](vendor/README.md) and
[`data/MANIFEST.yaml`](data/MANIFEST.yaml).

## What this repository establishes

Hybrid models allocate two caches with different scaling laws:

\[
C(L) = \frac{L M}{A L + G},
\]

where \(A\) is attention-KV bytes per token, \(G\) is recurrent-state bytes
per sequence, \(M\) is the cache budget, and \(L\) is context length. The
repository connects this continuous accounting model to vLLM's discrete block
and page allocation, then audits quality, serving stability, and a fail-closed
offline selector.

This is an allocator-characterization study. It does **not** introduce a new
quantizer, cache allocator, or online scheduler, and it does not claim a
workload-general end-to-end serving gain.

| Evidence | Supported result | Canonical artifact |
|---|---|---|
| Allocator matrix | bf16 recurrent state increases token capacity in all 52 paired cells; median gain 15.44% | [`capacity-phase-formal-corrected.analysis.json`](results/verified/2026-08-14/capacity-phase-formal-corrected.analysis.json) |
| Accounting model | state share and crossover recomputed from frozen parameters; median absolute residual 2.38%; signed residuals -5.16% to +13.21%, so the model is neither a lower nor an upper bound | [`capacity-2x2-analysis-corrected.json`](results/verified/2026-08-14/capacity-2x2-analysis-corrected.json) |
| GSM8K quality | state-only change -1.00 point, 95% CI [-2.04, +0.04]; int4-KV change -2.72 points, [-5.28, -0.17] | [`gsm8k-state9seed-v2-dependence-aware-20260814.json`](results/quality/gsm8k-state9seed-v2-dependence-aware-20260814.json) |
| Offline selector | the strict request selects full precision; medium, high, and negative-control requests have no feasible candidate | [`selector-audit.json`](results/verified/2026-08-14/controller-decisions/selector-audit.json) |
| Serving stability | the four-configuration temporal audit fails its primary gate: 183/720 continuous-goodput comparisons exceed 10% tolerance | [`m4_formal_analysis.json`](results/reproduction/2026-08-13/m4-four-config/analysis-r3/m4_formal_analysis.json) |

## Quick verification

Use Linux or WSL2 with Python 3.10-3.12. Native Windows vLLM execution is
outside the supported environment.

```bash
git clone https://github.com/TanC8186/hybrid-cache-precision.git
cd hybrid-cache-precision
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,dev]"
./reproduce.sh verify
```

The verification command runs the unit tests and independently checks 16
source hashes, eight vector-figure pairs, the corrected capacity source, the
two-way clustered GSM8K analysis, and the crossover and state-share values
quoted in Section III of the camera-ready. It does not require a GPU or
download model weights. The manuscript's own reanalysis of record ran on
Python 3.13.5 with NumPy 2.1.3, SciPy 1.15.3, and statsmodels 0.14.4; the
numbers reproduce on the supported 3.10-3.12 environment used here.

To regenerate the committed paper figures from the frozen aggregate evidence:

```bash
./reproduce.sh figures
```

Full GPU reproduction requires a 32 GB RTX 5090 environment, Qwen3.5 model
weights, licensed datasets, and the pinned vLLM patch stack. See
[`docs/artifact.md`](docs/artifact.md) before running it. Experiment stages
run through the per-study launchers listed there; the unified `make sweep` /
`bench` / `eval` / `analyze` targets are not implemented and exit with a
pointer to those launchers.

## Repository map

| Path | Purpose |
|---|---|
| `src/kvcache/` | Core quantization, cache, calibration, and selector code |
| `configs/` | Frozen model, dataset, precision, benchmark, and experiment contracts |
| `scripts/` | Experiment launchers, analyzers, controller tools, and figure builders |
| `tests/` | Unit and contract tests for the research code |
| `results/quality/` | Aggregated quality analyses |
| `results/verified/` | Hash-addressed validation and reproduction evidence |
| `results/reproduction/` | Temporal reruns and audit reports |
| `paper/dls2026/` | Camera-ready DLS 2026 manuscript and editable figure sources |
| `paper/mlsys2026/` | Earlier, longer MLSys-format manuscript of the same study |
| `vendor/vllm-patches/` | Reconstructable two-commit vLLM patch stack |
| `data/MANIFEST.yaml` | Dataset provenance, licenses, revisions, and checksums |

Raw model weights, datasets, local virtual environments, downloaded vLLM
sources, and large server archives are intentionally excluded. Aggregate
evidence and its checksums remain in Git.

## Artifact status

The DLS 2026 camera-ready is the version of record; `paper/mlsys2026/` is an
earlier, longer-format manuscript of the same study and is not a separate
replication. The archived release tag is `dls2026-camera-ready`; a persistent
archival identifier (DOI) is planned. Frozen experiment contracts preserve the
absolute paths of the original rented server as provenance; copy and adapt
them for a new machine rather than editing archived contracts in place.
