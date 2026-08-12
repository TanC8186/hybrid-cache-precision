## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: ANALYZED
- Version Label: joint_precision_calibration_validation_v1

## Validation Report

- **Source**: joint-precision-four-config-calibration-r1-20260811
- **Overall Confidence**: CAUTION
- **Gate**: PASS
- **Evidence Status**: ANALYZED

### Integrity

| Check | Result |
|---|---:|
| Launcher exit code | 0 |
| Requested / audited samples | 144 / 144 |
| Completed / expected requests | 320400 / 320400 |
| Failed requests | 0 |
| Silent exclusions | 0 |
| Server sessions | 4 |
| Aggregated cells | 48 |
| Profile rows | 240 |
| Raw SHA-256 sidecars verified by analyzer | 439 / 439 |

### Statistical Interpretation

- Unit of analysis: one independent workload trace / seed repeat.
- Each allocation-workload-rate cell uses three repeats and a Student-t 95% confidence interval with df=2.
- Requests are denominator evidence, not independent statistical repeats.
- The 240 pointwise SLO rows are calibration inputs, not confirmatory hypothesis tests.
- P values and multiple-comparison rejection decisions are therefore not used for paper claims at this stage.

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Calibration reuse | Seeds 7, 42, and 2026 construct the controller profile and cannot serve as independent confirmation. | All profile rows |
| Small repeat count | Each cell has n=3 independent repeats; t intervals are valid but imprecise. | CI bounds |
| Deployment scope | Measurements cover one GPU, one model, and the frozen rate grid. | Generalization |
| Evidence boundary | The profile may drive M2 confirmation, but calibration values are not paper-usable quantitative evidence. | Promotion |

### Fallacy Scan

- **Coverage**: 11/11

| Fallacy | Severity | Detail |
|---|---|---|
| Simpson's paradox | NOTE | Results remain stratified by allocation, workload, and offered rate; no pooled direction is used. |
| Ecological fallacy | NOTE | Inference is limited to seed/trace repeats; requests are not treated as independent repeats. |
| Berkson's paradox | CAUTION | The single GPU, model, and predeclared rate grid are selected deployment conditions. |
| Collider bias | NOTE | No post-treatment covariate adjustment or conditioned regression is performed. |
| Base rate neglect | NOTE | Every rate reports the full request denominator and failed-request fraction. |
| Regression to the mean | NOTE | Rates and seeds were frozen before execution rather than selected from extreme outcomes. |
| Survivorship bias | NOTE | The audit requires all 144 samples and retains every failed request as an SLO miss. |
| Look-elsewhere effect | CAUTION | The 240 pointwise SLO profile rows are calibration inputs, not claim-level significance tests. |
| Garden of forking paths | NOTE | Seeds, rates, thresholds, t critical, and aggregation rules are frozen in the contract. |
| Correlation != causation | NOTE | Allocation is experimentally controlled; claims remain bounded to this deployment matrix. |
| Reverse causality | NOTE | Precision configuration precedes each serving measurement through a cold-start deployment epoch. |

### Reproducibility

- **Determinism class**: environment-sensitive seeded serving benchmark
- **Method**: not run for calibration; independent seeds 11, 23, and 47 are reserved for M2 confirmation
- **Verdict**: CANNOT_VERIFY
- **Promotion**: profile construction and confirmatory execution authorized; paper quantitative use not authorized

### Raw Evidence

- Full source: `/root/autodl-tmp/controller-calibration-r1-20260811/joint-precision-four-config-calibration-r1-20260811` (1465063691 bytes)
- Compact archive: `/root/autodl-tmp/controller-calibration-r1-20260811/audit/compact-evidence.tar.gz`
- Compact archive SHA-256: `497edaf7bc290f9d27ee7c92c5dde019c8a7defd1c2d8e7d552fadeee405e2a3`
- Git stores contracts, analyses, statuses, hashes, launch/preflight evidence, and derived profile artifacts; request-level JSON remains on the server data disk.
