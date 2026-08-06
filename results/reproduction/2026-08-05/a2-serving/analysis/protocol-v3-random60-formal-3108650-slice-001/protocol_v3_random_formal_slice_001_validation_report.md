## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_protocol_v3_random_formal_slice_001_v1

## Validation Report

- **Attempt**: `a2-comparative-serving-random60-formal-v3-piecewise-3108650-westd-01`
- **Slice**: `slice-001`
- **Verdict**: `PROTOCOL_V3_RANDOM_FORMAL_SLICE_001_PASSED`
- **Partial Attempt**: `true`
- **Completed Samples**: `5/45`
- **Requests**: `9,600/9,600`
- **Failed Requests**: `0`

| Sample | Completed | Failed | P99 TTFT ms | Sustainable TTFT thresholds ms |
|---|---:|---:|---:|---|
| fp16__random__r30__s7 | 1,800 | 0 | 226.67 | 250, 500, 1000, 2000, 3000 |
| fp16__random__r30__s42 | 1,800 | 0 | 255.11 | 250, 500, 1000, 2000, 3000 |
| fp16__random__r30__s2026 | 1,800 | 0 | 224.25 | 250, 500, 1000, 2000, 3000 |
| fp16__random__r35__s7 | 2,100 | 0 | 323.09 | 500, 1000, 2000, 3000 |
| fp16__random__r35__s42 | 2,100 | 0 | 390.45 | 500, 1000, 2000, 3000 |

This slice is operational evidence only. The Random formal denominator remains
incomplete at 5/45 samples and cannot support a formal efficacy claim.
