## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_comparative_piecewise_serving_pilot_v1

## Validation Report

- **MVEx**: `a2-comparative-serving-mvex-piecewise-f7a79f5-westd-01`
- **Pilot**: `a2-comparative-serving-pilot-piecewise-f7a79f5-westd-02`
- **Pilot Verdict**: `PILOT_FAILED_CLIENT_FD_LIMIT_AND_SHAREGPT_WINDOW_BRACKETING`
- **MVEx Evidence Status**: `UNVERIFIED`
- **Pilot Evidence Status**: `QUARANTINED`
- **Overall A2 Status**: `PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING`

### Integrity

The comparative MVEx completed 9,000/
9,000 requests with zero failures. The pilot
published all 21 planned samples, but request-level accounting is
35,092 completed plus
8 failed out of
35,100. No request is silently excluded.

All six server sessions exited with return code zero, recorded no exception or
runtime-fatal signature, and prove `CUDAGraphMode.PIECEWISE`. The eight pilot
failures are confined to `fp16__random__r50__s7` and contain the client-side
`Too many open files` signature. The remote soft `nofile` limit was 1024.

### Pilot Results

| Allocation | Workload | Offered req/s | Completed | Failed | P99 TTFT ms | P99 TPOT ms | Drain s | Sustainable TTFT thresholds ms |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fp16 | random | 30 | 1,800 | 0 | 244.43 | 24.38 | 1.07 | 250, 500, 1000, 2000, 3000 |
| fp16 | random | 40 | 2,400 | 0 | 5257.98 | 49.29 | 6.50 | none |
| fp16 | random | 50 | 2,992 | 8 | 21554.76 | 49.49 | 22.82 | none |
| fp16 | sharegpt | 10 | 600 | 0 | 92.29 | 12.50 | 7.88 | none |
| fp16 | sharegpt | 15 | 900 | 0 | 108.55 | 14.96 | 6.46 | none |
| fp16 | sharegpt | 20 | 1,200 | 0 | 121.55 | 16.56 | 4.33 | none |
| fp16 | sharegpt | 30 | 1,800 | 0 | 162.83 | 23.21 | 10.54 | none |
| int4 | random | 30 | 1,800 | 0 | 277.67 | 26.21 | 1.33 | 500, 1000, 2000, 3000 |
| int4 | random | 40 | 2,400 | 0 | 708.19 | 44.56 | 2.13 | 1000, 2000, 3000 |
| int4 | random | 50 | 3,000 | 0 | 14334.48 | 44.78 | 15.86 | none |
| int4 | sharegpt | 10 | 600 | 0 | 98.72 | 14.86 | 11.07 | none |
| int4 | sharegpt | 15 | 900 | 0 | 109.12 | 17.72 | 9.17 | none |
| int4 | sharegpt | 20 | 1,200 | 0 | 141.42 | 21.18 | 6.42 | none |
| int4 | sharegpt | 30 | 1,800 | 0 | 181.79 | 26.10 | 14.41 | none |
| packed_per_layer | random | 30 | 1,800 | 0 | 262.04 | 24.64 | 1.27 | 250, 500, 1000, 2000, 3000 |
| packed_per_layer | random | 40 | 2,400 | 0 | 1643.28 | 45.84 | 3.01 | 2000, 3000 |
| packed_per_layer | random | 50 | 3,000 | 0 | 16246.18 | 46.12 | 17.75 | none |
| packed_per_layer | sharegpt | 10 | 600 | 0 | 97.84 | 14.18 | 10.92 | none |
| packed_per_layer | sharegpt | 15 | 900 | 0 | 114.94 | 19.42 | 8.97 | none |
| packed_per_layer | sharegpt | 20 | 1,200 | 0 | 130.73 | 20.55 | 6.10 | none |
| packed_per_layer | sharegpt | 30 | 1,800 | 0 | 181.35 | 25.25 | 13.88 | none |

### Gate 2 Failure

The frozen success criteria required 35,100 successful requests, zero failures,
and at least one sustainable plus one unsustainable ShareGPT point for every
allocation under at least one tested TTFT threshold. Neither condition holds.
ShareGPT at 10 req/s has low P99 latency but drains for 7.88--11.07 seconds
after the 60-second arrival window; because goodput uses total benchmark
duration, goodput/offered is only 0.844--0.884. Lowering the offered rate does
not remove this fixed-tail bias.

### Evidence Boundary

The pilot is useful as protocol and environment diagnostics, but it is
`ANALYZED/QUARANTINED` and contributes zero rows to formal efficacy evidence.
Formal expansion is blocked. The next linked MVEx must raise the inherited soft
file-descriptor limit and test a longer ShareGPT measurement window while
preserving the real completion-length distribution. A new pilot may start only
after that MVEx demonstrates both zero failures and a sustainable/unsustainable
ShareGPT bracket for all three allocations.

### Fallacy Scan

- **Coverage**: 11/11
