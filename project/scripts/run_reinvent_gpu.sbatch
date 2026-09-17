#!/bin/bash
#SBATCH --job-name=reinvent_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/reinvent_gen_%j.out
#SBATCH --error=logs/reinvent_gen_%j.err
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR/project"

PIPE_PY=/cluster/scratch/nbhatt04/conda/envs/polymer_ai/bin/python
REINV_VENV="$PROJECT_DIR/project/venvs/reinvent312"
SCORING_PY="$PIPE_PY"

echo "== host: $(hostname) =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PIPE_PY" -c "import torch; print('driver torch cuda:', torch.cuda.is_available())"

echo "== create REINVENT venv (python3.11) =="
"$PIPE_PY" -m venv --copies "$REINV_VENV"
"$REINV_VENV/bin/pip" install --upgrade pip wheel setuptools

echo "== install torch CUDA + REINVENT4 =="
"$REINV_VENV/bin/pip" install torch==2.12.0 torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu126
"$REINV_VENV/bin/pip" install \
    "git+https://github.com/MolecularAI/Reinvent4.git#egg=reinvent"
"$REINV_VENV/bin/reinvent" --version || true
"$REINV_VENV/bin/python" -c "import torch; print('reinvent torch cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== run REINVENT generation phase on GPU =="
REINVENT_BIN="$REINV_VENV/bin/reinvent" \
REINVENT_DEVICE="auto" \
SCORING_PYTHON="$SCORING_PY" \
"$PIPE_PY" workflows/run_all.py --only reinvent
echo "== reinvent phase done (rc=$?) =="

echo "== rerun downstream on collected library =="
"$PIPE_PY" workflows/run_all.py --only downstream library portfolio lock prospective loop audit reports structure
echo "EXIT_OK"
