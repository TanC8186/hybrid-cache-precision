## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_comparative_protocol_v3_pilot_v1

## Validation Report

- **Audited Component**: `random`
- **Verdict**: `PROTOCOL_V3_RANDOM_PILOT_COMPONENT_PASSED`
- **Evidence Status**: `UNVERIFIED`
- **Overall A2 Status**: `PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING`

### Pilot Results

| Workload | Allocation | Offered req/s | Completed | Failed | Throughput/offered | P99 TTFT ms | P99 TPOT ms | Sustainable TTFT thresholds ms |
|---|---|---:|---:|---:|---:|---:|---:|---|
| random | fp16 | 30 | 1,800 | 0 | 0.9846 | 238.50 | 19.74 | 250, 500, 1000, 2000, 3000 |
| random | fp16 | 40 | 2,400 | 0 | 0.9061 | 5041.06 | 49.17 | none |
| random | fp16 | 50 | 3,000 | 0 | 0.7229 | 21647.43 | 49.47 | none |
| random | int4 | 30 | 1,800 | 0 | 0.9789 | 239.11 | 24.33 | 250, 500, 1000, 2000, 3000 |
| random | int4 | 40 | 2,400 | 0 | 0.9670 | 711.68 | 44.65 | 1000, 2000, 3000 |
| random | int4 | 50 | 3,000 | 0 | 0.7910 | 14354.94 | 44.81 | none |
| random | packed_per_layer | 30 | 1,800 | 0 | 0.9794 | 242.53 | 22.67 | 250, 500, 1000, 2000, 3000 |
| random | packed_per_layer | 40 | 2,400 | 0 | 0.9558 | 1489.20 | 45.68 | 2000, 3000 |
| random | packed_per_layer | 50 | 3,000 | 0 | 0.7756 | 15931.28 | 45.82 | none |

### Gate Result

All audited pilot components pass request conservation, full rate-30/rate-50 bracketing, server integrity, and PIECEWISE proof.

### Evidence Boundary

Pilot rows remain `ANALYZED/UNVERIFIED` and cannot support paper efficacy
claims. Formal execution requires the complete Random plus ShareGPT pilot to
pass; historical failed pilots and every MVEx denominator remain excluded.

### Fallacy Scan

- **Coverage**: 11/11
