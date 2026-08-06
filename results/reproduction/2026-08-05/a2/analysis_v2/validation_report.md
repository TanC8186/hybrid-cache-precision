## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: VERIFIED
- Version Label: a2_protocol_v2_validation_v1

## Validation Report

- **Source**: `a2-repro-v2-suite-c7379f0-westd-03`
- **Verification Scope**: A2 runtime/capacity mechanism and capacity ratios
- **Verdict**: `REPRODUCIBLE`
- **Overall A2 Status**: `PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING`

### Capacity Confirmation

| Allocation | Original | Confirmed | Symmetric Difference | Protocol-v2 |
|---|---:|---:|---:|---|
| legacy | 705,604 | 706,560 | 0.135% | PASS |
| uniform | 2,736,947 | 2,740,224 | 0.120% | PASS |
| packed | 2,280,448 | 2,283,520 | 0.135% | PASS |

The confirmation suite exactly repeated all three `westd-02` capacities. It also
passed the prospectively frozen 1% capacity and 0.1% ratio tolerances.

### Ratio Confirmation

- packed / legacy: `3.231884`
- packed / uniform: `0.833333`

### Verification Checks

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
- [x] `suite_exit_zero`
- [x] `four_attempts_complete`
- [x] `capacity_within_1_percent`
- [x] `ratios_within_0_1_percent`
- [x] `ratio_gate_passed`
- [x] `exact_repeat_of_discovery_suite`
- [x] `generation_nonempty`

### Scope Boundary

Only the A2 runtime/capacity sub-scope is `VERIFIED`. Packed serving SLO and
quality evaluation remain pending, so the overall A2 method is not yet fully
verified for paper-wide claims.

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
| Garden of Forking Paths | SOLID | Protocol-v2 tolerances were frozen before westd-03, and westd-02 was excluded from the confirmatory verdict. |
| Correlation != Causation | CAUTION | The common block-count increase supports, but does not prove, the memory-profile drift explanation. |
| Reverse Causality | NOTE | Not applicable to the controlled cache-layout comparison. |
