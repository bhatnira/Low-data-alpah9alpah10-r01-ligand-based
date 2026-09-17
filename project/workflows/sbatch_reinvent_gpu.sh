#!/bin/bash
#SBATCH --job-name=reinvent_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=logs/reinvent_gen_%j.out
#SBATCH --error=logs/reinvent_gen_%j.err
#SBATCH --open-mode=append
set -euo pipefail
ROOT="/cluster/home/nbhatt04/Low-data-alpah9alpah10-r01-ligand-based/project"
cd "$ROOT"

PIPE_PY=/cluster/scratch/nbhatt04/conda/envs/polymer_ai/bin/python
REINV_BIN="$ROOT/venvs/reinvent312/bin/reinvent"

echo "== host: $(hostname) =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/cluster/scratch/nbhatt04/conda/envs/polymer_ai/lib \
  "$REINV_BIN" --version || true

echo "== REINVENT4 in-domain generation (GPU) =="
export REINVENT_BIN="$REINV_BIN"
export REINVENT_SCORING_PYTHON="$PIPE_PY"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/cluster/scratch/nbhatt04/conda/envs/polymer_ai/lib \
  "$PIPE_PY" workflows/run_all.py --only reinvent
echo "== reinvent rc=$? =="

echo "== downstream triage + portfolio + reports on in-domain library =="
"$PIPE_PY" workflows/run_all.py --only chemspace candidates downstream library lock prospective loop audit reports
echo "== done rc=$? =="
