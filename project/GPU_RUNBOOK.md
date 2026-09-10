# GPU Runbook — REINVENT4 generation phase

Everything a fresh agent on a CUDA machine needs to install REINVENT4, run the
generative phase, and leave the repository's artifacts **and docs** consistent.

The pipeline itself is honesty-bound: **never fabricate molecules**. If the
REINVENT binary is absent, the phase writes valid per-mode TOML configs and a
`NOT_RUN_BINARY_ABSENT` manifest entry — but no molecules. Real molecules can
only appear via the run described below.

---

## 0. What this phase produces (success criteria)

After a successful GPU run you must have, inside `project/`:

- `reinvent/generated_molecules.csv` — non-empty; one row per unique molecule,
  all `experimental_status == "untested"`.
- `reinvent/generation_plan.csv` — `status` column shows `READY_RUNNABLE`.
- `reinvent/reinvent_manifest.json` — `run_status: "RUN_COMPLETED"`,
  `reinvent_available: true`, `reinvent_binary` + `reinvent_version` filled,
  `n_generated > 0`.
- `reinvent/out/<mode>/` populated for the 5 modes
  (`de_novo`, `information_gain`, `local_analog`, `scaffold_hopping`,
  `taf_disrupting`) plus `local_analog_libinvent` (LibInvent prior).
- Downstream phases re-run cleanly: `chemspace`, `candidates`, `library`,
  `lock`, `prospective`, `loop`, `audit`, `reports`.
- Docs synced: no remaining `READY_PENDING_REINVENT_BINARY` / "binary absent" /
  "0 molecules" REINVENT claims (run `python3 scripts/sync_generated_docs.py`).

---

## 1. Prerequisites (check first)

- NVIDIA GPU with a recent driver (`nvidia-smi` works).
- Python **3.12** (REINVENT4 requires `>=3.11`; 3.12 has the best CUDA-wheel
  coverage). Both `python3.12` and `venv` must be available.
- `git`, `curl`, `shasum`.
- Network access to pypi.org and download.pytorch.org.

The pipeline venv (the one that runs `workflows/run_all.py`) needs `numpy`,
`pandas`, `rdkit`, `scikit-learn`, `joblib`. This is the interpreter configured
as `reinvent.scoring_python` and used by the frozen scoring bridge
`reinvent/score_lbm.py`.

---

## 2. Install REINVENT4 (separate venv, recommended)

REINVENT4 is **source-only** — the PyPI project name is `reinvent`, but the
published wheel requires the CUDA torch index (`torch==2.12.0`), so install
with the explicit torch index URL.

```bash
cd project

# 2a. pipeline venv (scoring bridge + run_all.py)
python3.12 -m venv venvs/pipeline312
venvs/pipeline312/bin/pip install --upgrade pip
venvs/pipeline312/bin/pip install numpy pandas rdkit scikit-learn joblib

# 2b. REINVENT4 venv (torch-CUDA + generator)
python3.12 -m venv venvs/reinvent312
venvs/reinvent312/bin/pip install --upgrade pip
# CUDA wheels: cu126 = CUDA 12.6 (adjust to your driver, e.g. cu121/cu124/cu128)
venvs/reinvent312/bin/pip install torch==2.12.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu126
venvs/reinvent312/bin/pip install \
    "git+https://github.com/MolecularAI/Reinvent4.git#egg=reinvent"
```

Verification:

```bash
venvs/reinvent312/bin/reinvent --version
# -> "Reinvent 4 <version> ... using PyTorch 2.12.0"
venvs/reinvent312/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True <GPU name>
venvs/pipeline312/bin/python -c "import sklearn, joblib, rdkit; print('bridge deps OK')"
```

> Shortcut (single venv): install Reinvent4 *and* sklearn/joblib into the same
> env, run the pipeline with that Python, and leave `reinvent.scoring_python`
> empty (it defaults to `sys.executable`). The config `binary` field then stays
> empty and `venvs/reinvent*/bin/reinvent` is still auto-detected.

---

## 3. Priors (download first on any fresh clone)

`project/priors/` is gitignored (large model weights), so on a fresh clone the
two priors will be missing. Fetch and sha256-verify them (checksums are pinned
in `configs/config.yaml` and re-verified at runtime by `src/reinvent.py`):

```bash
./download_priors.sh
```

Run with `--force` to re-download. Never commit priors.

---

## 4. Configure + run

`configs/config.yaml` → `reinvent`:

```yaml
reinvent:
  available: false          # auto-flag at runtime; do not edit
  device: auto              # auto = cuda when a GPU is present, else cpu
  binary: ""                # empty = auto-detect (env/PATH/venvs)
  scoring_python: ""        # empty = the interpreter running the pipeline
```

Set explicitly only if auto-detection fails:

```bash
export REINVENT_BIN="$PWD/venvs/reinvent312/bin/reinvent"   # or set reinvent.binary
```

Invocation style is auto-resolved (Reinvent4 console scripts take the config
directly; legacy v3 binaries use `reinvent <config>`), so the pipeline can call
the binary itself — no manual config editing.

Run the pipeline **twice** (phase 3b consumes phase-13 output, so run 1
populates, run 2 is the converged pass):

```bash
venvs/pipeline312/bin/python workflows/run_all.py
venvs/pipeline312/bin/python workflows/run_all.py
```

The `reinvent` phase writes TOMLs, runs generation per mode into
`reinvent/out/<mode>/` (logs: `reinvent/logs/<mode>.log`), collects
`*summary.csv` into `reinvent/generated_molecules.csv`, and updates
`reinvent_manifest.json`. If a mode fails, the run continues and the manifest
records `PARTIAL_RUN_FAILURES` with per-mode return codes (honest, not silent).

Optional: run only the affected phases after generation to save time:

```bash
venvs/pipeline312/bin/python workflows/run_all.py --only reinvent chemspace candidates library lock prospective loop audit reports
```

---

## 5. Post-run verification + doc sync

```bash
# 1) molecules exist and every row is honesty-marked as untested
venvs/pipeline312/bin/python - <<'PY'
import pandas as pd, json
df = pd.read_csv("reinvent/generated_molecules.csv")
m = json.load(open("reinvent/reinvent_manifest.json"))
print(df.shape, sorted(df.experimental_status.unique()))
print(m["run_status"], m["reinvent_available"], m["n_generated"])
PY

# 2) tests
venvs/pipeline312/bin/python -m pytest -q

# 3) flip REINVENT doc sentinels to the real numbers/status
python3 scripts/sync_generated_docs.py
# -> reports per file changed; exit 0 only if no stale sentinels remain

# 4) residual check (must print nothing)
grep -rn "PENDING_REINVENT_BINARY\|binary absent" docs grant README.md 2>/dev/null || true
```

`sync_generated_docs.py` refuses to touch docs while
`reinvent/generated_molecules.csv` is missing or empty — it only rewrites
"planned/absent/0 molecules" phrasing with the **actual** counts from the
manifest, and never adds claims of experimental activity.

---

## 6. Commit

```bash
git add -A
git commit -m "REINVENT4 GPU run: generated molecules + downstream refresh"
git push
```

Push `main` and `synth-route-prediction` if that branch was touched.

---

## 7. Honesty guardrails (do not violate)

- Every generated row stays `experimental_status: "untested"`.
- Docs must say molecules are *in silico generated, not assayed*. No claim of
  potency, TAF disruption, or clinical value can be derived from generation.
- Never edit `generated_molecules.csv` by hand.
- If generation yields 0 molecules or all modes fail, keep the
  `NOT_RUN_BINARY_ABSENT`/`RUN_FAILED` truth and do not fabricate CSVs.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `run_status: NOT_RUN_BINARY_ABSENT` | binary not found — re-check `reinvent --version`, `REINVENT_BIN`, venv name `venvs/reinvent*/bin/` |
| torch CUDA wheels fail to install | try `--index-url https://download.pytorch.org/whl/cu118` or cu124 matching your driver |
| ExternalProcess scoring errors in `<mode>.log` | `reinvent.scoring_python` points at an interpreter without sklearn/joblib — point it at `venvs/pipeline312/bin/python` |
| `RUN_FAILED` / `PARTIAL_RUN_FAILURES` | read `reinvent/logs/<mode>.log`; common cause is TOML schema drift — check against `Reinvent4/configs/PARAMS.md` |
| pipeline venv 3.14 without sklearn wheels | use python3.12 for `venvs/pipeline312` |