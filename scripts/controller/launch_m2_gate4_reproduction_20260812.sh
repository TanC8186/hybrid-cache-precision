#!/usr/bin/env bash
set -Eeuo pipefail

WORKTREE=/root/autodl-tmp/MLSys_M2_20260812
PYTHON=/root/autodl-tmp/MLSys_Research/.venv/bin/python
OUTPUT_ROOT=/root/autodl-tmp/m2-gate4-repro-20260812-r1
PROFILE=results/verified/2026-08-12/controller-profile/joint-precision-four-config-calibration-physical-r1-20260812/physical_calibration_profile.json
SERVING_CONFIG=configs/experiments/joint_precision_controller_2b.yaml
LAUNCH_ROOT="${OUTPUT_ROOT}/launch"

export PYTHONPATH="${WORKTREE}/src:${WORKTREE}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

guard_resources() {
    local system_avail_kb data_avail_kb gpu_used
    system_avail_kb=$(df -Pk / | awk 'NR == 2 {print $4}')
    data_avail_kb=$(df -Pk /root/autodl-tmp | awk 'NR == 2 {print $4}')
    gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if (( system_avail_kb < 8 * 1024 * 1024 )); then
        printf 'system disk safety threshold reached: %s KiB available\n' "${system_avail_kb}" >&2
        return 70
    fi
    if (( data_avail_kb < 10 * 1024 * 1024 )); then
        printf 'data disk safety threshold reached: %s KiB available\n' "${data_avail_kb}" >&2
        return 71
    fi
    if (( gpu_used > 512 )); then
        printf 'GPU is not idle: %s MiB used\n' "${gpu_used}" >&2
        return 72
    fi
}

run_attempt() {
    local request_id=$1
    local expected_allocation=$2
    local parent_attempt="joint-precision-m2-pilot-${request_id}-s11-23-47-20260812"
    local attempt_id="joint-precision-m2-gate4-repro-${request_id}-s11-23-47-20260812-r1"
    local request="configs/controller/requests/joint_precision_m2_${request_id}_20260812.json"
    local launch_dir="${LAUNCH_ROOT}/${attempt_id}"

    guard_resources
    if [[ -e "${OUTPUT_ROOT}/${attempt_id}" || -e "${launch_dir}" ]]; then
        printf 'refusing to overwrite attempt: %s\n' "${attempt_id}" >&2
        return 64
    fi
    mkdir -p "${launch_dir}"
    printf '%s\n' "${expected_allocation}" >"${launch_dir}/expected_allocation"
    printf '%s\n' "${parent_attempt}" >"${launch_dir}/parent_attempt"
    printf '%s\n' logical_only >"${launch_dir}/evidence_review_mode"
    date -Is >"${launch_dir}/started_at"

    set +e
    timeout --signal=TERM --kill-after=30s 7200 \
        "${PYTHON}" scripts/controller/run_joint_precision_controller.py \
        --profile "${PROFILE}" \
        --request "${request}" \
        --serving-config "${SERVING_CONFIG}" \
        --phase confirmatory \
        --attempt-id "${attempt_id}" \
        --parent-attempt "${parent_attempt}" \
        --output-root "${OUTPUT_ROOT}" \
        --seeds 11,23,47 \
        --evidence-review-mode logical_only \
        >"${launch_dir}/run.log" 2>&1
    local exit_code=$?
    set -e

    printf '%s\n' "${exit_code}" >"${launch_dir}/exit_code"
    date -Is >"${launch_dir}/finished_at"
    if (( exit_code != 0 )); then
        printf 'attempt failed: %s rc=%s\n' "${attempt_id}" "${exit_code}" >&2
        return "${exit_code}"
    fi
}

mkdir -p "${OUTPUT_ROOT}"
if [[ -n "$(git -C "${WORKTREE}" status --porcelain=v1)" ]]; then
    printf 'remote worktree is dirty\n' >&2
    exit 73
fi
cd "${WORKTREE}"

run_attempt strict full
run_attempt medium state_only
run_attempt high joint
date -Is >"${LAUNCH_ROOT}/reproduction_finished_at"
