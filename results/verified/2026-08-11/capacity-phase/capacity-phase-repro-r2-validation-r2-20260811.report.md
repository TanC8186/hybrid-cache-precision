## Material Passport

- Origin Skill: experiment-skill
- Origin Mode: validate
- Origin Date: 2026-08-11
- Verification Status: ANALYZED
- Version Label: capacity_phase_r2_validation_report_v1

## Validation Report

- **Source**: `capacity-phase-formal-20260811` -> `capacity-phase-repro-r2-20260811`
- **Validation Attempt**: `capacity-phase-repro-r2-validation-r2-20260811`
- **Overall Confidence**: CAUTION
- **Reproducibility Verdict**: PARTIALLY_REPRODUCIBLE
- **Promotion Status**: BLOCKED

### Completeness And Integrity

| Check | Observed | Status |
|---|---:|---|
| Formal exit code | 0 | PASS |
| Requested/completed cells | 112/112 | PASS |
| `[OK]` / `[FAIL]` / `[DONE]` | 112 / 0 / 1 | PASS |
| `[SKIP]` / `[RETRY]` | 0 / 0 | PASS |
| Cell JSON / SHA-256 sidecars | 112 / 112 | PASS |
| Missing or mismatched cell hashes | 0 | PASS |
| Core bf16-state direction pairs | 52/52 positive | PASS |

### Statistical Findings

| Metric | Parent | R2 | Difference | Status |
|---|---:|---:|---:|---|
| Maximum cell token symmetric relative difference | N/A | N/A | 1.411193% | WITHIN 2% |
| 2B, fp16 KV median state gain | 13.803158% | 13.801364% | 0.001794 pp | WITHIN 2 pp |
| 2B, int4 KV median state gain | 30.042518% | 30.042518% | 0.000000 pp | WITHIN 2 pp |
| 9B, fp16 KV median state gain | 9.303324% | 9.303324% | 0.000000 pp | WITHIN 2 pp |
| 9B, int4 KV median state gain | 26.851834% | 27.302925% | 0.451091 pp | WITHIN 2 pp |

The unit of analysis is one frozen configuration cell. This is an
environment-sensitive allocator benchmark, so timing fields are excluded.
There are no null-hypothesis tests, p-values, or stochastic replicate means;
confidence intervals and multiple-comparison p-value correction are therefore
not applicable. Effect magnitude is reported directly as paired capacity gain
and reproduction difference.

### Warnings

| Type | Detail | Affected Evidence |
|---|---|---|
| Provenance block | The old formal contract records a vLLM runtime hash set that cannot execute its own frozen probe command. | Promotion to `VERIFIED` |
| Invocation failure | The first frozen validator command failed before comparison because script-path execution could not import the top-level `scripts` package. The failure is preserved under a separate attempt. | Validation audit trail only |
| Environment scope | Results cover the recorded RTX 5090 runtime and do not establish cross-GPU, serving, quality, or TP claims. | Claim boundary |

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Finding | Recommendation |
|---|---|---|---|
| Simpson's paradox | NOTE | All 52 paired directions are positive; no aggregate/subgroup direction reversal was observed across model and KV strata. | Keep subgroup rows available with aggregate summaries. |
| Ecological fallacy | NOTE | The inference unit remains the configuration cell; no request-level or user-level claim is made. | Do not generalize cell-level capacity to request behavior. |
| Berkson's paradox | NOTE | The frozen 112-cell matrix completed without post hoc sample selection. | Preserve the exhaustive denominator. |
| Collider bias | NOTE | No adjusted regression or conditioned common-effect variable is used. | Not applicable to this paired deterministic comparison. |
| Base-rate neglect | NOTE | No diagnostic classification metric is reported. | Not applicable. |
| Regression to the mean | NOTE | This is not an extreme-selected pre/post design. | Not applicable. |
| Survivorship bias | NOTE | All 112 requested cells completed; no failed, skipped, or retried cell entered a reduced denominator. | Continue fail-closed accounting. |
| Look-elsewhere effect | NOTE | The matrix and tolerances were frozen before R2; all cells and all four group medians are reported. | Keep exploratory follow-ups separate. |
| Garden of forking paths | CAUTION | The formal command was frozen, but the first post-processing invocation failed and required a separately recorded module-entry retry. | Fix and test both CLI entry modes before R3; do not erase the failed invocation. |
| Correlation versus causation | NOTE | The result supports only the controlled allocator configuration contrast on this runtime, not end-to-end serving or quality causation. | Keep claims within the measured capacity mechanism. |
| Reverse causality | NOTE | State dtype is set before deterministic capacity construction; no reverse temporal pathway applies. | Not applicable beyond the stated mechanism. |

### Reproducibility

- **Method**: full environment-sensitive re-run; structure, dtypes, capacity
  directions, per-cell tokens, and group medians compared; timing excluded.
- **Comparison Result**: all 112 cells and 52 core pairs passed the frozen
  tolerances, with zero token, direction, or median-gain failures.
- **Verdict**: PARTIALLY_REPRODUCIBLE.
- **Evidence Status**: ANALYZED.

### Observation, Inference, Recommendation

- **Observation**: R2 is a complete, internally valid 112-cell formal result
  whose numerical comparison with the old formal passes every declared
  tolerance.
- **Inference**: The capacity pattern is numerically stable under the accurately
  recorded R2 runtime, but the defective provenance of the old parent prevents
  this comparison from satisfying Gate 4.
- **Recommendation**: Freeze R3 with R2 as its parent, use the accurately frozen
  runtime and a tested validator entry point, and promote only if R2-to-R3
  reproduction and the full Gate 4 audit pass.
