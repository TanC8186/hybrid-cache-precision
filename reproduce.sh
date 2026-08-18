#!/usr/bin/env bash
# Artifact entry point. Run from Linux or WSL2 with Python 3.10-3.12.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
MODE="${1:-verify}"

usage() {
  cat <<'EOF'
Usage: ./reproduce.sh [verify|figures|paper|help]

  verify   Run unit tests and verify manuscript evidence (default).
  figures  Regenerate vector paper figures from committed evidence.
  paper    Build the IEEE/DLS manuscript with latexmk.
EOF
}

case "$MODE" in
  verify)
    "$PYTHON_BIN" -m pytest -q "$ROOT/tests"
    "$PYTHON_BIN" "$ROOT/paper/mlsys2026/figures/verify_figure_data.py"
    ;;
  figures)
    (
      cd "$ROOT/paper/mlsys2026/figures/vector_redesign"
      "$PYTHON_BIN" make_vector_figures.py
    )
    "$PYTHON_BIN" "$ROOT/paper/mlsys2026/figures/verify_figure_data.py"
    ;;
  paper)
    make -C "$ROOT/paper/dls2026"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
