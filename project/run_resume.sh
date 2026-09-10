#!/usr/bin/env bash
# Resume: downstream phases (chemspace -> reports) on the 31,397-molecule
# REINVENT pass-2 set, then tests + doc sync + residual sentinel check.
#SBATCH --job-name=lbm-resume
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err
set -euo pipefail
cd /cluster/home/nbhatt04/Low-data-alpah9alpah10-r01-ligand-based/project
export LD_LIBRARY_PATH="$PWD/venvs/pipeline312/lib:$PWD/venvs/reinvent312/lib:${LD_LIBRARY_PATH:-}"
export MPLBACKEND=Agg
PIP="$PWD/venvs/pipeline312/bin/python"

echo "=== RESUME downstream phases (chemspace -> reports) ==="
$PIP workflows/run_all.py --only chemspace candidates library lock prospective synth_route loop audit reports

echo "=== TESTS ==="
$PIP -m pytest tests/ -q 2>&1

echo "=== DOC SYNC ==="
$PIP scripts/sync_generated_docs.py 2>&1

echo "=== RESIDUAL SENTINEL CHECK (must print nothing) ==="
grep -rIn "PENDING_REINVENT_BINARY\|binary absent" docs grant workflows src configs tests scripts 2>/dev/null --include=*.md --include=*.txt --include=*.py || true

echo "=== DONE ==="