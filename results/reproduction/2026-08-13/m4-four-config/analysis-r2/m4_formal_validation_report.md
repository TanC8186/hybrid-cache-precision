## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-13
- Verification Status: ANALYZED
- Version Label: m4_four_config_formal_validation_v1

## Validation Report

- Source: `m4-four-config-serving-formal-39739e0-20260813-r2`
- Overall Confidence: CAUTION
- Evidence Status: ANALYZED; not promoted before Gate 4

### Integrity

The frozen formal matrix passed logical Gate 3: 144/144 cells, 320400/320400 measurement requests, zero failed requests, zero partial artifacts, and four precision-verified stopped server sessions. The detached launch has incomplete provenance; the gap is retained as a warning and is not backfilled.

### Statistical Findings

Primary family: 180 paired goodput comparisons (3 alternatives x [5 Random + 7 ShareGPT rates] x 5 TTFT thresholds), with seed/trace as the repeat unit. BH-FDR q<0.05 survivors: 68.

| Metric | Comparison | Workload | Rate | Mean delta | 95% CI | p | BH q | dz |
|---|---|---|---:|---:|---|---:|---:|---:|
| goodput @ 250ms | joint - full | random | 30 | -0.6412 | [-1.1867, -0.0957] | 0.03694 | 0.06954 | -2.920028092817062 |
| goodput @ 500ms | joint - full | random | 30 | -0.1901 | [-0.2231, -0.1570] | 0.001633 | 0.01592 | -14.269078635075664 |
| goodput @ 1000ms | joint - full | random | 30 | -0.1901 | [-0.2231, -0.1570] | 0.001633 | 0.01592 | -14.269078635075664 |
| goodput @ 2000ms | joint - full | random | 30 | -0.1901 | [-0.2231, -0.1570] | 0.001633 | 0.01592 | -14.269078635075664 |
| goodput @ 3000ms | joint - full | random | 30 | -0.1901 | [-0.2231, -0.1570] | 0.001633 | 0.01592 | -14.269078635075664 |
| goodput @ 250ms | kv_only - full | random | 30 | -0.7122 | [-1.0638, -0.3606] | 0.01291 | 0.03829 | -5.0317303648112555 |
| goodput @ 500ms | kv_only - full | random | 30 | -0.2179 | [-0.2251, -0.2107] | 5.844e-05 | 0.00263 | -75.51799843649992 |
| goodput @ 1000ms | kv_only - full | random | 30 | -0.2179 | [-0.2251, -0.2107] | 5.844e-05 | 0.00263 | -75.51799843649992 |
| goodput @ 2000ms | kv_only - full | random | 30 | -0.2179 | [-0.2251, -0.2107] | 5.844e-05 | 0.00263 | -75.51799843649992 |
| goodput @ 3000ms | kv_only - full | random | 30 | -0.2179 | [-0.2251, -0.2107] | 5.844e-05 | 0.00263 | -75.51799843649992 |
| goodput @ 250ms | state_only - full | random | 30 | -0.0953 | [-0.2449, 0.0543] | 0.1112 | 0.154 | -1.5830730110091813 |
| goodput @ 500ms | state_only - full | random | 30 | -0.0133 | [-0.0436, 0.0171] | 0.2012 | 0.255 | -1.0842225488008608 |
| goodput @ 1000ms | state_only - full | random | 30 | -0.0133 | [-0.0436, 0.0171] | 0.2012 | 0.255 | -1.0842225488008608 |
| goodput @ 2000ms | state_only - full | random | 30 | -0.0133 | [-0.0436, 0.0171] | 0.2012 | 0.255 | -1.0842225488008608 |
| goodput @ 3000ms | state_only - full | random | 30 | -0.0133 | [-0.0436, 0.0171] | 0.2012 | 0.255 | -1.0842225488008608 |
| goodput @ 250ms | joint - full | random | 35 | -1.6562 | [-6.9954, 3.6830] | 0.3136 | 0.3442 | -0.7705631370130802 |
| goodput @ 500ms | joint - full | random | 35 | -0.1939 | [-0.2167, -0.1710] | 0.0007485 | 0.01592 | -21.090977163942327 |
| goodput @ 1000ms | joint - full | random | 35 | -0.1993 | [-0.2353, -0.1634] | 0.001754 | 0.01592 | -13.767327744155942 |
| goodput @ 2000ms | joint - full | random | 35 | -0.1993 | [-0.2353, -0.1634] | 0.001754 | 0.01592 | -13.767327744155942 |
| goodput @ 3000ms | joint - full | random | 35 | -0.1993 | [-0.2353, -0.1634] | 0.001754 | 0.01592 | -13.767327744155942 |
| goodput @ 250ms | kv_only - full | random | 35 | -4.5053 | [-7.6698, -1.3408] | 0.02563 | 0.06021 | -3.5366498760972855 |
| goodput @ 500ms | kv_only - full | random | 35 | -0.3210 | [-0.5955, -0.0465] | 0.03729 | 0.06954 | -2.9054006263944254 |
| goodput @ 1000ms | kv_only - full | random | 35 | -0.2885 | [-0.4977, -0.0793] | 0.02724 | 0.06021 | -3.426343214818681 |
| goodput @ 2000ms | kv_only - full | random | 35 | -0.2885 | [-0.4977, -0.0793] | 0.02724 | 0.06021 | -3.426343214818681 |

Only the first 24 primary rows are shown in this compact report; the complete comparison table is in `m4_formal_comparisons.csv`.

### Warnings

- n=3 paired repeats gives wide, assumption-sensitive CIs; request-level rows are not independent.
- Serving is environment-sensitive; latency and throughput should not be treated as exact deterministic quantities.
- Launcher provenance is incomplete, so this attempt remains ANALYZED pending a separate attempt with complete provenance.

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
| Garden of forking paths | CAUTION | The formal matrix is frozen, but launcher provenance is incomplete; the gap is disclosed rather than repaired retrospectively. |
| Correlation != causation | CAUTION | Allocation is controlled, but environment-sensitive serving measurements do not by themselves identify a mechanism or universal causal benefit. |
| Reverse causality | NOTE | Precision allocation is assigned before each cold-start serving epoch, so latency cannot determine the assignment retrospectively. |

### Reproducibility

- Method: pending separate-attempt environment-sensitive run-stability comparison
- Verdict: CANNOT_VERIFY
- Promotion: quantitative paper use is blocked until the new attempt passes structural and tolerance comparison.
