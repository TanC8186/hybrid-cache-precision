## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-04
- Verification Status: VERIFIED
- Version Label: e3_reproducibility_validation_v1

## Validation Report

- **Source**: `e3-v2-formal-d1d52c4-01`
- **Overall Confidence**: CAUTION
- **Reproducibility Verdict**: REPRODUCIBLE
- **Method**: environment-sensitive seeded re-run, 10% symmetric relative tolerance

### Integrity

| Attempt | Role | Samples | Requests | Failures | Arrival ratio | SHA files |
|---|---|---:|---:|---:|---:|---:|
| e3-v2-formal-d1d52c4-01 | formal | 72/72 | 160200/160200 | 0 | 0.999661--1.000427 | 665 |
| e3-v2-repro-random-d1d52c4-02 | random_reproduction | 18/18 | 43200/43200 | 0 | 0.999840--1.000292 | 176 |
| e3-v2-repro-sharegpt-d1d52c4-02 | sharegpt_reproduction | 24/24 | 39600/39600 | 0 | 0.999673--1.000239 | 229 |
| e3-v2-repro-sharegpt-upper-d1d52c4-01 | sharegpt_upper_neighbor | 6/6 | 14400/14400 | 0 | 0.999902--1.000183 | 70 |

### Reproducibility

- Cell means compared: 80
- Maximum symmetric relative difference: 4.993%
- Boundary comparisons: 60/60 exact
- ShareGPT rate-40 upper-neighbor samples: 6/6 unsustainable at all five TTFT thresholds.

### Sustainable Boundary

| Workload | TTFT | FP16 | INT4 | Relative change |
|---|---:|---:|---:|---:|
| random | 250 ms | 35.00 | 35.00 | +0.0% |
| random | 500 ms | 35.00 | 36.67 | +4.8% |
| random | 1000 ms | 35.00 | 40.00 | +14.3% |
| random | 2000 ms | 35.00 | 40.00 | +14.3% |
| random | 3000 ms | 35.00 | 40.00 | +14.3% |
| sharegpt | 250 ms | 28.33 | 23.33 | -17.6% |
| sharegpt | 500 ms | 28.33 | 23.33 | -17.6% |
| sharegpt | 1000 ms | 28.33 | 23.33 | -17.6% |
| sharegpt | 2000 ms | 28.33 | 23.33 | -17.6% |
| sharegpt | 3000 ms | 28.33 | 23.33 | -17.6% |

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | CAUTION | Random and ShareGPT effects have opposite signs; pooling workloads would be misleading. |
| Ecological fallacy | NOTE | Inference stays at the seed-by-cell level; requests are not treated as independent replicates. |
| Berkson's paradox | NOTE | No outcome-conditioned sample selection was used; workloads and rates were frozen in advance. |
| Collider bias | NOTE | No post-treatment control variable is used in the boundary comparison. |
| Base rate neglect | NOTE | Not applicable to the offered-rate and SLO-goodput measurements. |
| Regression to the mean | NOTE | Independent attempts reproduce the same discrete boundary; no pre/post extreme-group inference is made. |
| Survivorship bias | NOTE | All planned samples and requests are accounted for; failed historical attempts remain excluded and preserved. |
| Look-elsewhere effect | CAUTION | Five TTFT thresholds are evaluated; all are reported and no threshold is selected as uniquely confirmatory. |
| Garden of forking paths | CAUTION | Protocol v2 followed documented v1 failures; v1 attempts are quarantined and v2 gates were rerun before formal execution. |
| Correlation is not causation | CAUTION | The allocation comparison is controlled, but conclusions are limited to this model, stack, hardware, and workload. |
| Reverse causality | NOTE | Not applicable to the controlled serving configuration comparison. |

### Claim Boundary

- Random synthetic traffic reproduces a threshold-dependent INT4 capacity gain: none at 250 ms, small at 500 ms, and 14.3% at 1000--3000 ms.
- ShareGPT reverses direction: the mean sustainable boundary is 23.33 req/s for INT4 versus 28.33 req/s for FP16 (-17.6%).
- These results do not support a workload-general claim that INT4 increases SLO capacity.
- Confidence remains CAUTION because n=3, rates use a 5 req/s grid, and ShareGPT intervals are wide.
