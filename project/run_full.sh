#!/usr/bin/env bash
# Full LBM pipeline: pass 1 (populate) + pass 2 (converged), tests, doc sync.
set -euo pipefail
cd /cluster/home/nbhatt04/Low-data-alpah9alpah10-r01-ligand-based/project
export LD_LIBRARY_PATH="$PWD/venvs/pipeline312/lib:$PWD/venvs/reinvent312/lib:${LD_LIBRARY_PATH:-}"
export MPLBACKEND=Agg
PIP="$PWD/venvs/pipeline312/bin/python"

echo "=== PASS 1 (full pipeline, incl. REINVENT4 GPU generation) ==="
$PIP workflows/run_all.py
echo "=== PASS 2 (converged) ==="
$PIP workflows/run_all.py

echo "=== TESTS ==="
$PIP -m pytest tests/ -q 2>&1

echo "=== DOC SYNC ==="
$PIP scripts/sync_generated_docs.py 2>&1

echo "=== RESIDUAL SENTINEL CHECK (must print nothing) ==="
grep -rn "PENDING_REINVENT_BINARY\|binary absent" docs grant README.md . 2>/dev/null --include=*.md --include=*.txt --include=*.py || true

echo "=== DONE ==="