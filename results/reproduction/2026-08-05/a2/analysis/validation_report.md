## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_reproduction_validation_v1

## Validation Report

- **Source**: `a2-repro-suite-c7379f0-westd-02`
- **Overall Confidence**: CAUTION
- **Frozen-Contract Verdict**: `PARTIALLY_REPRODUCIBLE`
- **Environment-Sensitive Comparison**: `REPRODUCIBLE`
- **A2 Evidence Status**: `PASSED_NOT_VERIFIED`

### Capacity Comparison

| Allocation | Original | Reproduced | Relative Change | Exact Gate |
|---|---:|---:|---:|---|
| legacy | 705,604 | 706,560 | +0.135% | MISMATCH |
| uniform | 2,736,947 | 2,740,224 | +0.120% | MISMATCH |
| packed | 2,280,448 | 2,283,520 | +0.135% | MISMATCH |

The three capacity values remain within the generic 10% environment-sensitive
tolerance, but none exactly matches the predeclared cross-host gate. The
mechanism and ratio gates reproduce; A2 therefore remains `PASSED`, not
`VERIFIED`.

### Ratio Gates

- packed / legacy: `3.231884` (required >= 3.0)
- packed / uniform: `0.833333` (required 0.80--0.92)

### Structural Checks

- [x] `runtime_eight_of_eight`
- [x] `runtime_generation_nonempty`
- [x] `runtime_and_capacity_packed_agree`
- [x] `legacy_requested_map_and_flag`
- [x] `legacy_has_24_independent_groups`
- [x] `uniform_requested_map_and_flag`
- [x] `uniform_has_one_attention_group`
- [x] `packed_eight_of_eight`
- [x] `packed_has_mixed_attention_group`
- [x] `mamba_ssm_cache_dtype_float32`

### Fallacy Scan

- **Coverage**: 11/11

| Fallacy | Severity | Detail |
|---|---|---|
| Simpson's Paradox | NOTE | Not applicable; no subgroup aggregation. |
| Ecological Fallacy | NOTE | Not applicable; inference stays at the host/configuration level. |
| Berkson's Paradox | NOTE | Not applicable; no selected correlation sample. |
| Collider Bias | NOTE | Not applicable; no covariate adjustment. |
| Base Rate Neglect | NOTE | Not applicable; no diagnostic probabilities. |
| Regression to the Mean | NOTE | Not applicable; no extreme-score selection. |
| Survivorship Bias | SOLID | The failed suite is preserved and excluded; all four replacement attempts are reported. |
| Look-Elsewhere Effect | SOLID | All predeclared capacities and structural checks are reported. |
| Garden of Forking Paths | CAUTION | A transport environment variable was added only under a new linked suite; numerical criteria were not changed. |
| Correlation != Causation | CAUTION | The common block-count increase supports, but does not prove, the memory-profile drift explanation. |
| Reverse Causality | NOTE | Not applicable to the controlled cache-layout comparison. |
