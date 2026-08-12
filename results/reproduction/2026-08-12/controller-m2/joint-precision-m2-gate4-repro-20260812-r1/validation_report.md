## Material Passport

- Origin Skill: experiment-skill
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: VERIFIED
- Version Label: joint_precision_m2_gate4_validation_v1

## Validation Report

- **Source**: `joint-precision-m2-gate4-reproduction-20260812-r1`
- **Gate 4 Verdict**: `PASS`
- **Evidence Status**: `VERIFIED`
- **Reproducibility**: `REPRODUCIBLE`
- **Audit Mode**: logical review only; no SHA-256 or hash validation performed

### Integrity Findings

The reproduction contains 9/9 seeded samples, 18,000/18,000 measurement
requests, 1,080 declared warmup requests, zero failed requests, zero silent
exclusions, and three launcher exit codes of 0. Parent linkage, selector mapping,
sample membership, precision commands, server-log proofs, and requested 500/200
ms SLO attainment (3/3 seeds for every budget) match structurally.

| Budget | Selected | Metric | Parent | Reproduction | Symmetric relative diff | Tolerance | Status |
|---|---|---|---:|---:|---:|---:|---|
| strict | full | mean_goodput_req_s | 29.562961 | 29.553602 | 0.0317% | 10% | WITHIN_TOLERANCE |
| strict | full | mean_p95_ttft_ms | 194.491683 | 194.232241 | 0.1334% | 20% | WITHIN_TOLERANCE |
| strict | full | mean_p95_tpot_ms | 18.649691 | 18.039428 | 3.2722% | 20% | WITHIN_TOLERANCE |
| medium | state_only | mean_goodput_req_s | 29.563116 | 29.552274 | 0.0367% | 10% | WITHIN_TOLERANCE |
| medium | state_only | mean_p95_ttft_ms | 196.663769 | 192.281649 | 2.2282% | 20% | WITHIN_TOLERANCE |
| medium | state_only | mean_p95_tpot_ms | 17.939456 | 17.841375 | 0.5467% | 20% | WITHIN_TOLERANCE |
| high | joint | mean_goodput_req_s | 38.878163 | 38.905324 | 0.0698% | 10% | WITHIN_TOLERANCE |
| high | joint | mean_p95_ttft_ms | 377.040434 | 372.417479 | 1.2261% | 20% | WITHIN_TOLERANCE |
| high | joint | mean_p95_tpot_ms | 38.112705 | 38.769293 | 1.6936% | 20% | WITHIN_TOLERANCE |

### Statistical Scope

The experiment is classified as an environment-sensitive seeded serving
benchmark. Comparisons use the predeclared seed-level means and symmetric
relative differences. Individual requests are retained for denominator and SLO
logic checks but are not treated as independent statistical repeats.

### Fallacy Scan

- **Coverage**: 11/11

The Gate 2 fallacy scan remains applicable and is preserved in both logical
audits. The reproduction adds no post hoc endpoints or exclusions.

### Promotion Decision

Gate 4 passes: structural checks and all predeclared environment-sensitive metric tolerances pass. The scoped M2 selector slice is promoted to VERIFIED.

The verified scope remains one RTX 5090, Qwen3.5-2B, 4096 context, Random
workload, and TP=1. This result does not verify cross-model, cross-context,
cross-hardware, TP=2/4, or mechanism claims.
