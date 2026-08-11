## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-11
- Verification Status: ANALYZED
- Version Label: joint_precision_four_config_mvex_r2_validation_v1
- Integrity Pass Date: 2026-08-11T13:15:17Z

## Validation Report

- **Source**: `joint-precision-four-config-mvex-r2-20260811`
- **Gate 1 Verdict**: `PASS`
- **Evidence Status**: `UNVERIFIED` diagnostic evidence
- **Overall Confidence**: `CAUTION`
- **Reproducibility**: `CANNOT_VERIFY` at single-seed MVEx

### Integrity Findings

The frozen set is complete: 4/4 samples, 7,200/7,200 measurement requests,
480/480 declared warmup requests, zero failed requests, zero missing or extra
sample directories, and launcher exit code 0. All 19
raw SHA-256 sidecars match. Every allocation used one distinct cold-start server
session, passed its exact state/KV/PIECEWISE log proofs, remained healthy after
benchmarking, and stopped with return code 0.

| Allocation | Requests | Throughput req/s | P95 TTFT ms | P95 TPOT ms | Goodput @ TTFT 250 ms |
|---|---:|---:|---:|---:|---:|
| full | 1800/1800 | 29.529 | 192.66 | 17.80 | 29.496 |
| kv_only | 1800/1800 | 29.351 | 205.73 | 22.70 | 29.171 |
| state_only | 1800/1800 | 29.538 | 189.36 | 17.35 | 29.505 |
| joint | 1800/1800 | 29.368 | 218.59 | 23.62 | 29.058 |

### Warnings

- The first full-precision cold start logged stale Triton cubin reload warnings after cache cleanup; vLLM rebuilt/loaded artifacts, reached health, completed the sample, and passed the post-benchmark health check.
- The vLLM controlled shutdown path logs EngineDeadError/force-kill cleanup noise, but each server status records exception=null and returncode=0 after a healthy sample.

### Fallacy Scan

- **Coverage**: 11/11

This MVEx is a pipeline and denominator gate, not an effect-estimation study.
With one seed, one synthetic workload, and one feasible rate, no confidence
interval, p value, multiple-comparison decision, controller advantage, or
paper-level quantitative claim is valid. Calibration must use the frozen
multi-seed/rate matrix before profile construction.

### Promotion Decision

Gate 1 passes and authorizes the predeclared calibration slice. The run artifact
remains `UNVERIFIED`; it is not upgraded to `VERIFIED` because no applicable
reproducibility re-run or multi-seed statistical comparison has been completed.
