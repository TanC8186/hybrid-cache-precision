# A2 Remote Server Cleanup Audit

## Material Passport

- Origin Skill: experiment-skill
- Origin Mode: run
- Origin Date: 2026-08-11
- Verification Status: VERIFIED
- Version Label: a2_remote_cleanup_audit_v1

## Scope

- Remote host: `autodl-container-kmzddlce45-c57bf976`
- Remote source: `/root/autodl-tmp/a2-serving-20260805-f7a79f5`
- Local archive: `results/reproduction/2026-08-05/a2-serving`
- Cleanup purpose: remove a server-side duplicate only after local-content coverage is proven.

## Pre-Deletion Inventory

- Remote regular files: 6,827
- Remote apparent bytes: 33,564,817,799
- Local regular files: 9,325
- Local bytes: 37,001,571,550
- Distinct local SHA-256 values: 2,294

## Content Verification

- Full remote manifest: `a2-serving-remote-full-manifest-20260811.sha256`
- Full remote manifest SHA-256: `6233bf161727a857b463f2fc461322af8805bce3e785fe1aea13eb5ec98784ff`
- Remote files already covered by local content hashes: 6,769 / 6,827
- Remaining remote-only files before residual capture: 58
- Remaining categories: builder/watcher logs, boundary snapshots, monitor/orchestrator metadata, and one packaging tool.
- Residual archive: `../raw/a2-server-residual-20260811.tar.gz`
- Residual archive entries: 250
- Residual archive bytes: 112,402
- Residual archive SHA-256: `08eb2a55cee4232508c880bf5fce8b50d87071f0adffb007e28380b67afb11f3`

The residual archive contains the complete source categories containing all 58
previously uncovered files. The downloaded archive hash matches the server-side
sidecar. Together, the existing local content and residual archive cover the
entire remote file set; the full manifest preserves the remote path-to-hash map.

## Decision

The remote tree is a verified duplicate and may be deleted after an immediate
open-file check, atomic rename into the fixed quarantine path, and revalidation
of the 6,827-file / 33,564,817,799-byte inventory. This audit concerns archive
integrity only and does not change the evidence status of the underlying A2
experiment results.

## Completion

- Completed at: `2026-08-11T19:48:43+08:00`
- Open process references before rename: 0
- Quarantine inventory: 6,827 files / 33,564,817,799 apparent bytes
- Deleted remote source: `/root/autodl-tmp/a2-serving-20260805-f7a79f5`
- Data-disk free space after deletion: 74 GB
- Server summary: `a2-remote-cleanup-summary-20260811.txt`
