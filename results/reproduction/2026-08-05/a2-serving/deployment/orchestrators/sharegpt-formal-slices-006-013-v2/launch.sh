#!/usr/bin/env bash
set -euo pipefail

export START_SLICE=6
export END_SLICE=13
export ORCH_NAME=sharegpt-formal-slices-006-013-v2

exec bash ./run-impl.sh
