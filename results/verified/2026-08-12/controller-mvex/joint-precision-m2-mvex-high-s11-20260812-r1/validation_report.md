## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: ANALYZED
- Version Label: joint_precision_m2_mvex_validation_v1

## Validation Report

- **Source**: `joint-precision-m2-mvex-high-s11-20260812-r1`
- **Gate 1 Verdict**: `PASS`
- **Evidence Status**: `UNVERIFIED` diagnostic evidence
- **Overall Confidence**: `CAUTION`
- **Reproducibility**: `CANNOT_VERIFY` at single-seed MVEx

### Integrity Findings

The scoped slice contains 1/1 validated sample and 2,400/2,400 measurement
requests, with zero failed requests, zero silent exclusions, one cold-start
server session, and internally consistent contracts/results. The selector selected `joint`
and the executed command proves int4 KV, bfloat16 state, and PIECEWISE graphs.

| Metric | Value |
|---|---:|
| Request throughput | 39.069372 req/s |
| Throughput / offered | 0.976734 |
| P95 TTFT | 332.048225 ms |
| P99 TTFT | 409.034306 ms |
| P95 TPOT | 39.677307 ms |
| P99 TPOT | 40.134665 ms |
| Goodput at 500/200 ms | 39.069372 req/s |

### Warnings

- This is a minimum viable execution gate, not a statistical comparison.
- The outer SSH wrapper recorded a nonnumeric shell status token; controller and runner artifacts independently record successful completion and return code 0.
- The frozen M2 contract covers Qwen3.5-2B, 4096 context, and Random workload only; it does not establish the full multi-model/context claim.

### Fallacy Scan

- **Coverage**: 11/11

- **Simpson's paradox** (`NOTE`): The MVEx has one predeclared allocation, workload, rate, and seed; no effect claim is promoted.
- **Ecological fallacy** (`NOTE`): The MVEx has one predeclared allocation, workload, rate, and seed; no effect claim is promoted.
- **Berkson's paradox** (`NOTE`): The MVEx has one predeclared allocation, workload, rate, and seed; no effect claim is promoted.
- **Collider bias** (`NOTE`): The MVEx has one predeclared allocation, workload, rate, and seed; no effect claim is promoted.
- **Base rate neglect** (`NOTE`): The MVEx has one predeclared allocation, workload, rate, and seed; no effect claim is promoted.
- **Regression to the mean** (`NOTE`): The MVEx has one predeclared allocation, workload, rate, and seed; no effect claim is promoted.
- **Survivorship bias** (`NOTE`): The MVEx has one predeclared allocation, workload, rate, and seed; no effect claim is promoted.
- **Look-elsewhere effect** (`NOTE`): The MVEx has one predeclared allocation, workload, rate, and seed; no effect claim is promoted.
- **Garden of forking paths** (`NOTE`): The MVEx has one predeclared allocation, workload, rate, and seed; no effect claim is promoted.
- **Correlation is not causation** (`NOTE`): The MVEx has one predeclared allocation, workload, rate, and seed; no effect claim is promoted.
- **Reverse causality** (`NOTE`): The MVEx has one predeclared allocation, workload, rate, and seed; no effect claim is promoted.

### Promotion Decision

Gate 1 passes and authorizes the predeclared pilot. This artifact remains
`UNVERIFIED`; it is not paper-usable until the multi-seed pilot, independent
reproducibility comparison, and statistical audit pass.
