# Results and evidence

This directory stores auditable research evidence, not an undifferentiated
dump of every run.

| Path | Meaning |
|---|---|
| `quality/` | Aggregated PPL, retrieval, reasoning, and dependence-aware quality analyses |
| `verified/` | Frozen contracts, validated analyses, hashes, and reproduction evidence organized by date |
| `reproduction/` | Temporal reruns, cross-run comparisons, and validation reports |
| `ablations/` | Exploratory and screening results; not automatically paper evidence |

Evidence labels are semantic:

- `SCREEN`: search or triage evidence only.
- `PILOT`: protocol validation or effect-size estimation.
- `ANALYZED`: completed analysis that has not passed its reproduction gate.
- `VERIFIED`: a frozen artifact that passed its declared validation contract.
- `FAILED_GATE`: a valid negative audit result; retain it without promoting its
  headline claim.

## Current canonical set

- Capacity: `verified/2026-08-14/capacity-*-corrected*`
- Controller: `verified/2026-08-14/controller-decisions/` and
  `verified/2026-08-14/controller-profile/`
- GSM8K: `quality/*dependence-aware-20260814.json`
- Serving stability: `reproduction/2026-08-13/m4-four-config/analysis-r3/`

Run the manuscript ledger to verify the exact files and values consumed by the
paper:

```bash
python paper/mlsys2026/figures/verify_figure_data.py
```

New formal outputs must be produced by a tracked analyzer and accompanied by
their contract and SHA-256 sidecar. Large request-level traces, server dumps,
model outputs, and archives stay in ignored local storage unless a specific
claim cannot be audited without them.
