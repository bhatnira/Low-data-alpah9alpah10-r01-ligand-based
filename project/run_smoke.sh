#!/usr/bin/env bash
set -euo pipefail
cd /cluster/home/nbhatt04/Low-data-alpah9alpah10-r01-ligand-based/project
PIP="$PWD/venvs/pipeline312/bin/python"
RV="$PWD/venvs/reinvent312"
export LD_LIBRARY_PATH="$RV/lib:$PWD/venvs/pipeline312/lib:${LD_LIBRARY_PATH:-}"
export REINVENT_BIN="$RV/bin/reinvent"
echo "NODE: $(hostname)"; nvidia-smi -L
$PIP -c "import torch" 2>/dev/null && echo "torch in pipeline (unexpected)" || echo "pipeline has no torch (OK)"
$RV/bin/python -c "import torch; print('REINVENT venv cuda:', torch.cuda.is_available())"
$PIP workflows/run_all.py --only reinvent --config configs/config.smoke.yaml 2>&1
