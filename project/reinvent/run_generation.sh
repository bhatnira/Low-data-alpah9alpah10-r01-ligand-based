#!/usr/bin/env bash
# Run REINVENT4 staged-learning for every generated per-mode config.
# Usage: REINVENT_BIN=/path/to/reinvent ./reinvent/run_generation.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMAND="${REINVENT_BIN:-}"
if [ -z "$COMMAND" ]; then
  for cand in reinvent reinvent4; do
    if command -v "$cand" >/dev/null 2>&1; then COMMAND="$(command -v "$cand")"; break; fi
  done
fi
if [ -z "$COMMAND" ]; then
  echo "REINVENT4 binary not found. Set REINVENT_BIN or install REINVENT4." >&2
  exit 1
fi
for toml in "$ROOT"/reinvent/configs/*_staged_learning.toml; do
  mode="$(basename "$toml" | sed 's/_staged_learning\.toml//')"
  echo "==> $mode  (${toml##*/})"
  outdir="$ROOT/reinvent/out/$mode"
  mkdir -p "$outdir"
  (cd "$outdir" && "$COMMAND" reinvent "$toml")
done
