#!/usr/bin/env bash
# Run REINVENT staged-learning for every generated per-mode config.
# Usage: REINVENT_BIN=/path/to/reinvent ./reinvent/run_generation.sh
# Invocation style is resolved per binary:
#   - Reinvent4 console script / reInvent / reinvent4:  <bin> <config.toml>
#   - legacy Reinvent v3 CLI:                            <bin> reinvent <config.toml>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMAND="${REINVENT_BIN:-}"
if [ -z "$COMMAND" ]; then
  for cand in reinvent reinvent4 reInvent; do
    if command -v "$cand" >/dev/null 2>&1; then COMMAND="$(command -v "$cand")"; break; fi
  done
fi
if [ -z "$COMMAND" ]; then
  echo "REINVENT binary not found. Set REINVENT_BIN or install REINVENT4." >&2
  exit 1
fi
BASE="$(basename "$COMMAND")"
for toml in "$ROOT"/reinvent/configs/*_staged_learning.toml; do
  mode="$(basename "$toml" | sed 's/_staged_learning\.toml//')"
  echo "==> $mode  (${toml##*/})"
  outdir="$ROOT/reinvent/out/$mode"
  mkdir -p "$outdir"
  ARGS=()
  case "$BASE" in
    reinvent)
      if "$COMMAND" --version 2>/dev/null | grep -qi 'reinvent 4'; then
        ARGS=()
      else
        ARGS=( reinvent )
      fi ;;
    reinvent4|reInvent) ARGS=() ;;
    *) ARGS=() ;;
  esac
  (cd "$outdir" && "$COMMAND" "${ARGS[@]}" "$toml")
done
