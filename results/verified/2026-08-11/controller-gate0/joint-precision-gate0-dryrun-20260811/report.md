## Material Passport

- Origin Skill: experiment-skill
- Origin Mode: run
- Origin Date: 2026-08-11
- Verification Status: UNVERIFIED
- Version Label: joint_precision_controller_gate0_dryrun_v1

## Experiment Result

- **ID**: `joint-precision-gate0-dryrun-20260811`
- **Type**: generic controller preflight
- **Status**: completed
- **Phase**: Gate 0 dry-run
- **Parent Attempt**: none
- **Code Revision**: `6ad20b4bcaf1003d947e53a9dd49c381db7c76b7`
- **Working Directory**: `/root/autodl-tmp/MLSys_Controller_1037a6c`
- **Output Directory**: `/root/autodl-tmp/controller-gate0-20260811/joint-precision-gate0-dryrun-20260811`
- **Artifact Timestamp**: `2026-08-11 18:32:38 +08:00`
- **Exit Code**: `0`
- **Hard Timeout**: 300 seconds

Frozen command:

```text
PYTHONPATH=/root/autodl-tmp/MLSys_Controller_1037a6c/src:/root/autodl-tmp/MLSys_Controller_1037a6c PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=30s 300 /root/autodl-tmp/MLSys_Research/.venv/bin/python scripts/controller/run_joint_precision_controller.py --profile configs/controller/fixtures/joint_precision_gate0_profile.json --request configs/controller/fixtures/joint_precision_gate0_request.json --serving-config configs/experiments/joint_precision_controller_2b.yaml --phase gate0 --attempt-id joint-precision-gate0-dryrun-20260811 --output-root /root/autodl-tmp/controller-gate0-20260811 --seeds 7 --dry-run
```

The command was executed from the recorded working directory after sourcing
`/etc/profile.d/mlsys-data-disk.sh`.

## Preflight Evidence

| Check | Result |
|---|---|
| Dedicated sparse worktree | PASS; restored from the index and clean at final revision |
| Focused tests | PASS; 32/32 on the remote host |
| Full tests | PASS; 122/122 on the remote host |
| Ruff lint | PASS |
| Ruff format check | PASS; 11 files checked |
| System/data disk | PASS; 19 GiB / 17 GiB available before collection |
| GPU use | None; 0 MiB and 0% after completion, as expected for `--dry-run` |
| Attempt collision | PASS; destination did not exist before launch |

The shared virtual environment was editable-installed against the older
`/root/autodl-tmp/MLSys_Research/src` path. The first diagnostic test collection
therefore failed with `ModuleNotFoundError` before executing tests. The final
preflight and frozen command explicitly bind `PYTHONPATH` to the clean controller
worktree. This environment correction occurred before the formal dry-run.

## Output Files

| File | Bytes | SHA-256 |
|---|---:|---|
| `decision.json` | 2,270 | `f1c69ef55ff4eb006eed529f0bcd47b8dd565e7cb89c08d85237c28826d0be67` |
| `controller_contract.json` | 3,520 | `0143e0a262cbc63b1b8553de4078fc863bd1a116cbd4b73c8df1e3a535823a6f` |
| `controller_result.json` | 1,480 | `62e98f1fc1f4253d3aed589bfa05fc0ea82e7d7d7e09a3fe642ae4acf88e517b` |

Each JSON has an adjacent raw-digest `.sha256` sidecar. The three remote hashes,
downloaded sidecars, and independent local hashes match exactly. No failure file
or unexpected artifact is present.

## Output Summary

| Field | Observed | Status |
|---|---|---|
| Controller result | `DRY_RUN_VALIDATED` | PASS |
| Profile status | `TEST_FIXTURE` | PASS for dry-run only |
| Selected configuration | `joint` | PASS |
| KV dtype | `int4_per_token_head` | PASS |
| Recurrent-state dtype | `bfloat16` | PASS |
| Deployment transition | cold restart required | PASS |
| Planned sample count | 1 | PASS |
| Planned sample ID | `joint__random__r40__s7` | PASS |
| Decision latency | 0.178635 ms | Recorded diagnostic only |
| Root Git state | clean, exact revision match | PASS |

The selected mapping expands to the exact vLLM arguments
`--kv-cache-dtype int4_per_token_head` and
`--mamba-ssm-cache-dtype bfloat16`. The result mapping and sample plan are
structurally identical to the frozen controller contract.

## Anomalies Detected

The first command-line SSH transport was closed before authentication and before
the remote command started. A read-only audit proved that the attempt directory
was absent and no controller process existed. The same frozen command was then
sent once through a Paramiko transport. This was a transport non-launch event,
not a crashed or retried experiment attempt, and no denominator was created or
discarded.

## Evidence Boundary

This artifact proves only that the strict selector, evidence hashes, request
matching, deployment mapping, and one-sample runner plan agree end to end. It
does not start vLLM, measure quality, latency, goodput, capacity utilization, or
controller efficacy. The fixture profile is intentionally blocked from real
execution. Real GPU execution remains gated on a hash-verified, non-fixture
calibration profile built from four-configuration measurements.

No statistical test, confidence interval, multiple-comparison decision, or
paper-level quantitative claim is made from this deterministic dry-run. The
11-category statistical fallacy audit applies to the later calibration/formal
evidence, not to this non-measurement preflight artifact.
