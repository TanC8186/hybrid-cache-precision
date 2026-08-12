## Material Passport

- Origin Skill: experiment-skill
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: ANALYZED
- Version Label: joint_precision_m2_pilot_logical_audit_v1

## Validation Report

- **Source**: `joint-precision-selector-m2-20260812`
- **Gate 2 Verdict**: `PASS`
- **Evidence Status**: `ANALYZED`
- **Overall Confidence**: `CAUTION`
- **Reproducibility**: `CANNOT_VERIFY` until the separate Gate 4 attempt
- **Audit Mode**: logical review only; no SHA-256 or hash validation performed

### Integrity Findings

The frozen pilot is complete: 9/9 seeded samples, 18,000/18,000 measurement
requests, 1,080 declared warmup requests, zero failed requests, zero missing or
extra samples, zero NaNs, zero residual partial artifacts, and three launcher
exit codes of 0. The selector logic independently reproduces all three mappings:
`strict -> full`, `medium -> state_only`, and `high -> joint`. Server commands
and logs prove the matching KV/state precision and PIECEWISE graph mode.

The confidence intervals below use the three seeds as the independent units
(Student-t, df=2). Individual requests are not treated as independent repeats.

| Budget | Selected | SLO seeds | Goodput req/s, mean [95% CI] | Mean P95 TTFT ms | Mean P95 TPOT ms | Required/max concurrency |
|---|---|---:|---:|---:|---:|---:|
| strict | full | 3/3 | 29.563 [29.523, 29.603] | 194.49 | 18.65 | 0.926 |
| medium | state_only | 3/3 | 29.563 [29.533, 29.594] | 196.66 | 17.94 | 0.963 |
| high | joint | 3/3 | 38.878 [38.263, 39.493] | 377.04 | 38.11 | 0.840 |

### Statistical Scope

- The requested 500 ms TTFT / 200 ms TPOT SLO is the primary endpoint.
- Calibration lower confidence bounds are guardrails, not point predictions;
  the report therefore records signed confirmatory-minus-LCB residuals rather
  than mislabeling them as ordinary prediction errors.
- With n=3 seeds per budget, intervals are descriptive and wide. No p value,
  equivalence claim, multi-model generalization, or mechanism attribution is promoted.

### Fallacy Scan

- **Coverage**: 11/11

- **Simpson's paradox** (`NOTE`): Each budget/allocation cell remains separate; no pooled direction replaces cell-level results.
- **Ecological fallacy** (`NOTE`): The inferential unit is the seeded benchmark sample, not an individual request.
- **Berkson's paradox** (`CAUTION`): The pilot intentionally covers one GPU, one model/context, Random workload, and three selected budget strata.
- **Collider bias** (`NOTE`): No post-treatment covariate adjustment or conditioned regression is used.
- **Base-rate neglect** (`NOTE`): Every sample retains its full request denominator and failures count as SLO misses.
- **Regression to the mean** (`NOTE`): Budget requests, rates, and confirmatory seeds were frozen before pilot execution.
- **Survivorship bias** (`NOTE`): All nine planned samples and all 18,000 measurement requests are required; none are silently excluded.
- **Look-elsewhere effect** (`CAUTION`): The requested 500/200 ms SLO is primary; the remaining frozen TTFT sweep is descriptive.
- **Garden of forking paths** (`NOTE`): Mappings, seeds, rates, denominators, and SLO thresholds were frozen in the package contract.
- **Correlation is not causation** (`CAUTION`): The selector executes controlled precision configurations, but the scoped pilot cannot establish cross-model or mechanism claims.
- **Reverse causality** (`NOTE`): Each precision decision precedes a cold-start deployment epoch and its measurements.

### Promotion Decision

Gate 2 passes. This pilot is `ANALYZED`, not `VERIFIED`, and is not yet
paper-usable quantitative evidence. Promotion requires a separately identified
Gate 4 reproduction and comparison under the declared environment-sensitive
tolerances.
