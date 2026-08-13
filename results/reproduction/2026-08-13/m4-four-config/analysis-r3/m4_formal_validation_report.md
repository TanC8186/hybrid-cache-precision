## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-13
- Verification Status: ANALYZED
- Version Label: m4_four_config_formal_validation_v1

## Validation Report

- Source: `m4-four-config-serving-repro-39739e0-20260813-r3`
- Overall Confidence: CAUTION
- Evidence Status: ANALYZED; not promoted before Gate 4

### Integrity

The frozen formal matrix passed logical Gate 3: 144/144 cells, 320400/320400 measurement requests, zero failed requests, zero partial artifacts, and four precision-verified stopped server sessions. Launcher provenance is complete and records exit code 0.

### Statistical Findings

Primary family: 180 paired goodput comparisons (3 alternatives x [5 Random + 7 ShareGPT rates] x 5 TTFT thresholds), with seed/trace as the repeat unit. BH-FDR q<0.05 survivors: 73.

| Metric | Comparison | Workload | Rate | Mean delta | 95% CI | p | BH q | dz |
|---|---|---|---:|---:|---|---:|---:|---:|
| goodput @ 250ms | joint - full | random | 30 | -1.3702 | [-2.3176, -0.4229] | 0.02486 | 0.05812 | -3.5929163221570906 |
| goodput @ 500ms | joint - full | random | 30 | -0.2472 | [-0.2641, -0.2302] | 0.0002539 | 0.005079 | -36.22358681496198 |
| goodput @ 1000ms | joint - full | random | 30 | -0.2472 | [-0.2641, -0.2302] | 0.0002539 | 0.005079 | -36.22358681496198 |
| goodput @ 2000ms | joint - full | random | 30 | -0.2472 | [-0.2641, -0.2302] | 0.0002539 | 0.005079 | -36.22358681496198 |
| goodput @ 3000ms | joint - full | random | 30 | -0.2472 | [-0.2641, -0.2302] | 0.0002539 | 0.005079 | -36.22358681496198 |
| goodput @ 250ms | kv_only - full | random | 30 | -0.3812 | [-0.5523, -0.2101] | 0.01071 | 0.04102 | -5.533462143966171 |
| goodput @ 500ms | kv_only - full | random | 30 | -0.1526 | [-0.1993, -0.1059] | 0.005015 | 0.02099 | -8.12171567992883 |
| goodput @ 1000ms | kv_only - full | random | 30 | -0.1526 | [-0.1993, -0.1059] | 0.005015 | 0.02099 | -8.12171567992883 |
| goodput @ 2000ms | kv_only - full | random | 30 | -0.1526 | [-0.1993, -0.1059] | 0.005015 | 0.02099 | -8.12171567992883 |
| goodput @ 3000ms | kv_only - full | random | 30 | -0.1526 | [-0.1993, -0.1059] | 0.005015 | 0.02099 | -8.12171567992883 |
| goodput @ 250ms | state_only - full | random | 30 | -0.0176 | [-0.2048, 0.1695] | 0.7246 | 0.7453 | -0.23390103518215583 |
| goodput @ 500ms | state_only - full | random | 30 | 0.0098 | [-0.0013, 0.0208] | 0.06339 | 0.1047 | 2.182693030507949 |
| goodput @ 1000ms | state_only - full | random | 30 | 0.0098 | [-0.0013, 0.0208] | 0.06339 | 0.1047 | 2.182693030507949 |
| goodput @ 2000ms | state_only - full | random | 30 | 0.0098 | [-0.0013, 0.0208] | 0.06339 | 0.1047 | 2.182693030507949 |
| goodput @ 3000ms | state_only - full | random | 30 | 0.0098 | [-0.0013, 0.0208] | 0.06339 | 0.1047 | 2.182693030507949 |
| goodput @ 250ms | joint - full | random | 35 | -3.8795 | [-7.5975, -0.1614] | 0.0462 | 0.08663 | -2.5919579634604943 |
| goodput @ 500ms | joint - full | random | 35 | -0.2715 | [-0.3529, -0.1902] | 0.004816 | 0.02099 | -8.289823017752758 |
| goodput @ 1000ms | joint - full | random | 35 | -0.2825 | [-0.3510, -0.2139] | 0.003165 | 0.01676 | -10.237756089132407 |
| goodput @ 2000ms | joint - full | random | 35 | -0.2825 | [-0.3510, -0.2139] | 0.003165 | 0.01676 | -10.237756089132407 |
| goodput @ 3000ms | joint - full | random | 35 | -0.2825 | [-0.3510, -0.2139] | 0.003165 | 0.01676 | -10.237756089132407 |
| goodput @ 250ms | kv_only - full | random | 35 | -0.5545 | [-3.3646, 2.2556] | 0.4853 | 0.5049 | -0.4901811333983823 |
| goodput @ 500ms | kv_only - full | random | 35 | -0.1795 | [-0.2523, -0.1067] | 0.00877 | 0.03432 | -6.124550686669845 |
| goodput @ 1000ms | kv_only - full | random | 35 | -0.1904 | [-0.2445, -0.1363] | 0.004338 | 0.02055 | -8.737048180132941 |
| goodput @ 2000ms | kv_only - full | random | 35 | -0.1904 | [-0.2445, -0.1363] | 0.004338 | 0.02055 | -8.737048180132941 |

Only the first 24 primary rows are shown in this compact report; the complete comparison table is in `m4_formal_comparisons.csv`.

### Warnings

- n=3 paired repeats gives wide, assumption-sensitive CIs; request-level rows are not independent.
- Serving is environment-sensitive; latency and throughput should not be treated as exact deterministic quantities.
- Launcher provenance is complete; this individual attempt remains ANALYZED until the separate-attempt comparison is evaluated.

### Fallacy Scan

- Coverage: 11/11 checked

| Fallacy | Severity | Detail |
|---|---|---|
| Simpson's paradox | NOTE | All primary comparisons stay stratified by workload and offered rate; pooled directions are not used as the sole result. |
| Ecological fallacy | CAUTION | The inferential unit is the paired seed/trace cell, not an individual request; request counts only form cell metrics. |
| Berkson's paradox | CAUTION | The evidence is conditional on one model, one RTX 5090, and the frozen load grid; external deployment prevalence is not inferred. |
| Collider bias | NOTE | No post-treatment queue, latency, or success variable is conditioned on when forming paired differences. |
| Base rate neglect | NOTE | All 320,400 offered measurement requests remain in denominators and failures are counted as SLO misses. |
| Regression to the mean | NOTE | Confirmatory seeds 11/23/47 were frozen before this run and were not selected by observed outcomes. |
| Survivorship bias | NOTE | All 144 cells are retained and the audit fails closed on missing or failed cells; observed failure count is zero. |
| Look-elsewhere effect | CAUTION | 180 primary goodput tests plus secondary metrics and five thresholds are reported; BH-FDR is applied within the declared primary family. |
| Garden of forking paths | CAUTION | The frozen formal matrix and complete launcher provenance constrain researcher degrees of freedom. |
| Correlation != causation | CAUTION | Allocation is controlled, but environment-sensitive serving measurements do not by themselves identify a mechanism or universal causal benefit. |
| Reverse causality | NOTE | Precision allocation is assigned before each cold-start serving epoch, so latency cannot determine the assignment retrospectively. |

### Reproducibility

- Method: pending separate-attempt environment-sensitive run-stability comparison
- Verdict: CANNOT_VERIFY
- Promotion: quantitative paper use is blocked until the new attempt passes structural and tolerance comparison.
