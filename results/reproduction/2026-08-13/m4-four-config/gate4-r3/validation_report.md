## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-13
- Verification Status: ANALYZED
- Version Label: m4_four_config_gate4_stability_v1

## Validation Report

- Source: `m4-four-config-serving-formal-39739e0-20260813-r2` and `m4-four-config-serving-repro-39739e0-20260813-r3`
- Overall Confidence: CAUTION
- Reproducibility Verdict: NOT_REPRODUCIBLE
- Method: same-seed environment-sensitive second formal run; not independent replication

### Integrity

Both attempts contain 144/144 validated cells and 320400/320400 completed requests with zero failures. The rerun has complete launcher provenance with exit code 0, and all eight server sessions pass realized precision-log checks.

### Run Stability

- Cell-threshold comparisons within the predeclared 10% tolerance: 537/720
- Maximum symmetric relative difference: 79.610%
- Exact sustainable point labels: 713/720
- Exact all-seed boundaries: 38/40
- Distribution: {'<1%': 346, '1-5%': 159, '5-10%': 32, '>=10%': 183}

### Fallacy Scan

- Coverage: 11/11 checked

| Fallacy | Severity | Detail |
|---|---|---|
| Simpson's paradox | NOTE | Run stability is compared per allocation, workload, rate, seed, and threshold; no pooled direction is substituted. |
| Ecological fallacy | CAUTION | The comparison unit is a seeded serving cell; requests remain constituents of a cell metric, not independent repeats. |
| Berkson's paradox | NOTE | All frozen cells are required and no cell is selected by outcome. |
| Collider bias | NOTE | No post-treatment queue or latency variable is conditioned on in the stability comparison. |
| Base rate neglect | NOTE | Both attempts retain all offered requests and count failures as SLO misses. |
| Regression to the mean | CAUTION | The rerun uses the same frozen seeds; it is run stability, not a new independent sample. |
| Survivorship bias | NOTE | Any missing, failed, or incomplete cell blocks verification instead of being omitted. |
| Look-elsewhere effect | NOTE | The 10% cell-goodput tolerance and complete five-threshold family were frozen before inspecting r3. |
| Garden of forking paths | NOTE | The rerun command, matrix, tolerance, environment class, and timing-metric exclusion were frozen before completion. |
| Correlation != causation | CAUTION | Passing run stability supports measurement repeatability in this environment, not a universal mechanism claim. |
| Reverse causality | NOTE | Allocation is assigned before each cold-start serving session and cannot be selected by observed performance. |

### Claim Boundary

- Call this same-seed run stability, not an independent replication or additional n.
- Do not pool original and rerun request denominators or seed counts.
- Do not compare wall-clock duration as a deterministic reproducibility metric.
- Restrict quantitative claims to this model, GPU, software revision, workload, and offered-load grid.
