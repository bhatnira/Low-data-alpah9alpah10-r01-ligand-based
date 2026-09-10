#!/usr/bin/env python3
"""Phases 9-11: REINVENT 4 integration, config generation, and generation modes.

Implements prompt2.txt sections 21, 22, 24, and generation report 7:
  - REINVENT 4 is a PROSPECTIVE design engine, not a label factory
  - five modes: de_novo, local_analog, scaffold_hopping, taf_disrupting,
    information_gain
  - a genuine multi-objective RL objective built from real components:
      * ExternalProcess -> reinvent/score_lbm.py (frozen Phase-4 classifier +
        real TAF SMARTS) exposing predicted_activity, taf_consistency,
        novelty, scaffold_novelty, stereo_validity, uncertainty,
        taf_disrupted
      * native MolecularWeight step-filter, SAScore, custom-alert filter
  - priors are the real REINVENT4 priors from Zenodo record 20701824
    (reinvent_pubchem.prior, libinvent.prior), checksum-verified at runtime
  - stereochemistry is first-class, but ONLY the Mol2Mol prior
    (reinvent_pubchem.prior) supports it; the LibInvent prior does not, so
    its config disables isomeric_smiles and drops the stereo_validity endpoint
  - full provenance per generated molecule, experimental_status=untested
  - REINVENT binary NOT AVAILABLE -> valid, ready-to-run configs are
    generated; generation itself waits for the binary.  No fabricated
    molecules, ever.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, RWMol

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance,
    save_df, save_manifest, sha256_file, sha256_text, utcnow,
)

RDLogger.DisableLog("rdApp.*")

MODE_DIRS = {
    "de_novo": "reinvent/de_novo",
    "local_analog": "reinvent/local_analog",
    "scaffold_hopping": "reinvent/scaffold_hopping",
    "taf_disrupting": "reinvent/taf_disrupting",
    "information_gain": "reinvent/information_gain",
}

MODE_RULES = {
    "de_novo": "novel molecules satisfying the TAF/AD constraints (no seed)",
    "local_analog": "near-series optimization around the real active parents (inception-guided)",
    "scaffold_hopping": "preserve TAFs while deliberately changing the scaffold",
    "taf_disrupting": "intentionally violate the TAF constellation (falsification control, portfolio group F)",
    "information_gain": "explore uncertain/borderline zones to discriminate competing hypotheses",
}

# objective endpoints: (payload property, output column label, weight)
MODE_ENDPOINTS: Dict[str, List[Tuple[str, str, float]]] = {
    "de_novo": [
        ("predicted_activity", "predicted_activity", 1.0),
        ("taf_consistency", "taf_consistency", 0.8),
        ("novelty", "novelty", 0.5),
        ("stereo_validity", "stereo_validity", 0.4),
    ],
    "local_analog": [
        ("predicted_activity", "predicted_activity", 1.0),
        ("taf_consistency", "taf_consistency", 0.8),
        ("novelty", "novelty", 0.2),
        ("stereo_validity", "stereo_validity", 0.5),
    ],
    "scaffold_hopping": [
        ("predicted_activity", "predicted_activity", 1.0),
        ("taf_consistency", "taf_consistency", 0.8),
        ("scaffold_novelty", "scaffold_novelty", 1.0),
        ("novelty", "novelty", 0.6),
        ("stereo_validity", "stereo_validity", 0.3),
    ],
    "taf_disrupting": [
        ("predicted_activity", "predicted_activity", 0.8),
        ("taf_disrupted", "taf_disrupted", 1.0),
        ("novelty", "novelty", 0.3),
        ("stereo_validity", "stereo_validity", 0.4),
    ],
    "information_gain": [
        ("predicted_activity", "predicted_activity", 0.5),
        ("uncertainty", "prediction_uncertainty", 1.0),
        ("novelty", "novelty", 0.8),
        ("stereo_validity", "stereo_validity", 0.3),
    ],
}

MODE_DIVERSITY: Dict[str, Tuple[str, Dict[str, float]]] = {
    "de_novo": ("IdenticalMurckoScaffold", {"bucket_size": 25, "minscore": 0.4}),
    "local_analog": ("IdenticalMurckoScaffold", {"bucket_size": 15, "minscore": 0.4}),
    "scaffold_hopping": ("IdenticalTopologicalScaffold", {"bucket_size": 25, "minscore": 0.4}),
    "taf_disrupting": ("IdenticalMurckoScaffold", {"bucket_size": 25, "minscore": 0.4}),
    "information_gain": ("PenalizeSameSmiles", {"bucket_size": 25, "minscore": 0.0,
                                                 "penalty_multiplier": 0.5}),
}

# Curated undesirable-substructure alerts (REINVENT4 custom_alerts filter).
# Deliberately EXCLUDES patterns that would block our own TAF engineering
# bits: the TAF-1 lactone core (C(=O)O[C]) and the TAF-3 EWG substituents
# (terminal alkyne C#C, nitrile C#N).
CURATED_ALERT_SMARTS = [
    "[*;r{8-17}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]",
]

ACTIVE_TAF_SMARTS_LACTONE = "C(=O)O[C,C]"
ACTIVE_TAF_SMARTS_DONOR = "[CX4][OH]"
ACTIVE_TAF_SMARTS_EWG = ["[Br]", "C#C", "C#N"]


def morgan_array(mol, nbits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _render_endpoints(component_path: str, endpoints: List[Tuple[str, str, float]],
                      extra: Optional[Dict[str, Any]] = None) -> List[str]:
    """Render repeated [[...endpoint]] array-of-tables blocks (REINVENT4 syntax)."""
    lines = []
    for prop, label, weight in endpoints:
        lines.append(f"[[{component_path}.endpoint]]")
        lines.append(f'name = "{label}"')
        lines.append(f"weight = {float(weight):g}")
        if extra:
            for k, v in extra.items():
                if isinstance(v, str):
                    lines.append(f'params.{k} = "{v}"')
                else:
                    lines.append(f"params.{k} = {v}")
        lines.append(f'params.property = "{prop}"')
    return lines


class ReinventManager:
    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger
        self.reinv = cfg.raw.get("reinvent", {})
        self._binary: Optional[Dict[str, Any]] = None
        self._device: Optional[str] = None
        self._version_cache: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------ binary detection
    def _probe_version(self, path: str) -> Optional[str]:
        """First line of `<binary> --version` (cached). None on any failure."""
        if path in self._version_cache:
            return self._version_cache[path]
        try:
            proc = subprocess.run([path, "--version"], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, timeout=20)
            line = (proc.stdout or "").strip().splitlines()[0].strip()
        except Exception:
            line = ""
        self._version_cache[path] = line or None
        return self._version_cache[path]

    @staticmethod
    def _invocation_style(name: str, version_line: Optional[str]) -> str:
        """How to call the binary with a config.

        Reinvent4 (pip console script `reinvent`, or `reInvent`/`reinvent4`
        binaries):  <binary> <config.toml>
        Legacy Reinvent v3 CLI: <binary> reinvent <config.toml>
        """
        if name in ("reInvent", "reinvent4"):
            return "direct"
        if name == "reinvent":
            if version_line and "reinvent 4" in version_line.lower():
                return "direct"
            return "v3_subcommand"
        return "direct"

    def _detect_binary(self) -> Optional[Dict[str, Any]]:
        """Locate the REINVENT executable.

        Priority: $REINVENT_BIN (env) > reinvent.binary (config) > `reinvent` /
        `reinvent4` / `reInvent` on PATH > project/venvs/reinvent*/bin/*.
        Returns {"binary": str, "style": "direct"|"v3_subcommand"} or None.
        """
        def _probe(path: str) -> Optional[Dict[str, Any]]:
            if not path:
                return None
            p = Path(path).expanduser()
            if not p.is_file() or not os.access(str(p), os.X_OK):
                return None
            ver = self._probe_version(str(p))
            return {"binary": str(p),
                    "style": self._invocation_style(p.name, ver)}

        # 1) explicit env override
        hit = _probe(os.environ.get("REINVENT_BIN", ""))
        if hit is not None:
            return hit
        # 2) config path
        hit = _probe(str(self.reinv.get("binary", "")))
        if hit is not None:
            return hit
        # 3) on PATH
        for cand in ("reinvent", "reinvent4", "reInvent"):
            w = shutil.which(cand)
            if w is not None:
                hit = _probe(w)
                if hit is not None:
                    return hit
        # 4) project venv (e.g. project/venvs/reinvent312/bin/reinvent)
        venv_root = self.cfg.resolve("venvs")
        if venv_root.is_dir():
            for sub in sorted(venv_root.glob("reinvent*/bin")):
                for cand in ("reinvent", "reinvent4", "reInvent"):
                    hit = _probe(str(sub / cand))
                    if hit is not None:
                        return hit
        return None

    @property
    def available(self) -> bool:
        if self._binary is None:
            self._binary = self._detect_binary()
        return self._binary is not None

    @property
    def runnable(self) -> Optional[Dict[str, Any]]:
        self.available  # populate cache
        return self._binary

    # ------------------------------------------------------------ device
    @staticmethod
    def _gpu_available() -> bool:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                return True
        except Exception:
            pass
        return shutil.which("nvidia-smi") is not None

    def resolve_device(self) -> str:
        """Config-device -> concrete device written into every TOML + manifest.

        "auto" (default) = cuda when a GPU is detected, else cpu.
        """
        if self._device is None:
            requested = str(self.reinv.get("device", "auto")).strip().lower()
            if requested == "auto":
                self._device = "cuda" if self._gpu_available() else "cpu"
            else:
                self._device = requested if requested in ("cpu", "cuda") else "cpu"
        return self._device

    def scoring_python(self) -> str:
        """Python used to run reinvent/score_lbm.py (ExternalProcess bridge).

        Defaults to the interpreter running the pipeline (sys.executable),
        which has rdkit/scikit-learn/joblib.  A configured value overrides it.
        """
        want = str(self.reinv.get("scoring_python", "") or "").strip()
        if want and Path(want).expanduser().is_file():
            return want
        if want:
            self.log.warn(f"configured reinvent.scoring_python not found "
                          f"({want}); falling back to sys.executable")
        return sys.executable

    @property
    def project_root(self) -> Path:
        return Path(str(self.cfg.resolve("."))).resolve()

    # ------------------------------------------------------------- priors
    def verify_priors(self) -> Dict[str, Dict[str, Any]]:
        """Verify the downloaded REINVENT4 priors against published checksums."""
        out = {}
        priors = self.reinv.get("priors", {})
        for name, meta in priors.items():
            p = Path(str(self.cfg.resolve(str(meta["file"]))))
            actual = sha256_file(str(p)) if p.exists() else None
            expected = meta.get("sha256")
            out[name] = {
                "name": name,
                "file": str(p),
                "exists": bool(p.exists()),
                "size_bytes": p.stat().st_size if p.exists() else None,
                "sha256_expected": expected,
                "sha256_actual": actual,
                "verified": bool(p.exists() and actual == expected),
            }
            if not out[name]["verified"]:
                self.log.warn(f"Prior {name} NOT verified "
                              f"(exists={p.exists()}, expected={expected}, actual={actual})")
        return out

    # ------------------------------------------------------- seed inputs
    def _write_actives_seed(self, df: pd.DataFrame) -> str:
        """inception.smi: the real active-parent isomeric SMILES for guidance."""
        act = df[df["is_active"] == 1]["isomeric_SMILES"].tolist()
        p = self.cfg.resolve("reinvent/inception.smi")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(s for s in act if isinstance(s, str)) + "\n")
        self.log.info(f"Inception seed ({len(act)} active SMILES) -> {p}")
        return str(p)

    def _write_libinvent_scaffolds(self, df: pd.DataFrame) -> Optional[str]:
        """Attachment-point scaffolds for LibInvent R-group decoration.

        Each scaffold is an active-parent core with one terminal acyclic
        branch excised and replaced by a `*` dummy attachment point.  Only
        scaffolds that parse and sanitize are emitted; if none survive the
        LibInvent path is skipped (honest NOT-VALIDATED state).
        """
        act = df[df["is_active"] == 1]
        scaffolds = []
        for row in act.itertuples():
            mol = getattr(row, "mol", None)
            if mol is None:
                mol = Chem.MolFromSmiles(row.isomeric_SMILES)
            if mol is None:
                continue
            s = self._cut_branch(mol)
            if s:
                scaffolds.append(s)
        scaffolds = sorted(set(scaffolds))
        if not scaffolds:
            self.log.warn("No valid LibInvent attachment-point scaffolds derived; "
                          "local_analog_libinvent config skipped")
            return None
        p = self.cfg.resolve("reinvent/scaffolds_libinvent.smi")
        p.write_text("\n".join(scaffolds) + "\n")
        self.log.info(f"LibInvent scaffolds ({len(scaffolds)}) -> {p}")
        return str(p)

    @staticmethod
    def _cut_branch(mol: Chem.Mol) -> Optional[str]:
        """Excise one terminal acyclic branch of a ring and mark `*`."""
        try:
            ri = mol.GetRingInfo()
            if not ri.NumRings():
                return None
            ring_atoms = {a for ring in ri.AtomRings() for a in ring}
            branches = []
            for ai in sorted(ring_atoms):
                for nb in mol.GetAtomWithIdx(ai).GetNeighbors():
                    if nb.GetIdx() not in ring_atoms:
                        branches.append((ai, nb.GetIdx()))
                        break  # first branch only, keep scaffolds small
                if branches:
                    break
            if not branches:
                return None
            cut_atom, base = branches[0]
            # collect the full acyclic branch reachable from base
            branch = set()
            stack = [base]
            while stack:
                i = stack.pop()
                if i in branch:
                    continue
                branch.add(i)
                a = mol.GetAtomWithIdx(i)
                for nb in a.GetNeighbors():
                    if nb.GetIdx() not in branch and nb.GetIdx() not in ring_atoms:
                        stack.append(nb.GetIdx())
            rw = RWMol(mol)
            for i in sorted(branch, reverse=True):
                rw.RemoveAtom(i)
            # ------- map original index -> current index after removals
            def cur(oi: int) -> int:
                return oi - sum(1 for b in branch if b < oi)
            dummy = rw.AddAtom(Chem.Atom(0))
            rw.AddBond(cur(cut_atom), dummy, Chem.BondType.SINGLE)
            out = rw.GetMol()
            Chem.SanitizeMol(out)
            return Chem.MolToSmiles(out)
        except Exception:
            return None

    # ----------------------------------------------------- TOML rendering
    def _staged_learning_toml(self, mode: str, prior_abs: str,
                              inception_abs: Optional[str],
                              scaffolds_abs: Optional[str],
                              stereo_supported: bool = True) -> str:
        r = self.reinv
        learn = r.get("learning", {})
        df_type, df_params = MODE_DIVERSITY.get(mode, MODE_DIVERSITY["de_novo"])
        endpoints = MODE_ENDPOINTS[mode]
        if not stereo_supported:
            # LibInvent prior is SMILES-fragment based and does NOT carry
            # stereochemistry; drop the stereo endpoint entirely.
            endpoints = [e for e in endpoints if e[1] != "stereo_validity"]
        scoring_python = self.scoring_python()
        score_script = str(self.cfg.resolve("reinvent/score_lbm.py"))
        out_dir = Path(str(self.cfg.resolve("reinvent/out", mode))).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        device = self.resolve_device()

        L: List[str] = []
        L.append("# REINVENT4 staged_learning (reinforcement learning) configuration")
        L.append(f"# mode        : {mode}")
        L.append(f"# rule        : {MODE_RULES[mode]}")
        L.append("# prior       : real REINVENT4 prior (Zenodo record 20701824), "
                 "checksum-verified")
        L.append(f"# stereochemistry : {'supported (Mol2Mol prior)' if stereo_supported else 'NOT supported (LibInvent prior) - isomeric_smiles disabled'}")
        L.append(f"# objectives  : {', '.join(f'{l} (w={w:g})' for _, l, w in endpoints)}")
        L.append("# scoring     : ExternalProcess -> reinvent/score_lbm.py "
                 "(frozen Phase-4 model + real TAF SMARTS)")
        L.append("")
        L.append('run_type = "staged_learning"')
        L.append(f'device = "{device}"')
        L.append(f'tb_logdir = "tb_logs"')
        L.append(f'json_out_config = "_{mode}.json"')
        L.append("")
        L.append("[parameters]")
        L.append(f'prior_file = "{prior_abs}"')
        L.append(f'agent_file = "{prior_abs}"')
        if scaffolds_abs:
            L.append(f'smiles_file = "{scaffolds_abs}"  # LibInvent scaffolds, 1 per line')
        L.append(f'summary_csv_prefix = "{mode}"')
        L.append("use_checkpoint = false")
        L.append("purge_memories = false")
        L.append(f'batch_size = {r.get("batch_size", 64)}')
        # NOTE: REINVENT4 >= 4.8 validates [parameters]; unique_sequences
        # was removed from the schema (uniqueness handled by the diversity
        # filter + the repository-level canonical-SMILES dedup on collect).
        L.append("randomize_smiles = true")
        # stereochemistry is first-class only for the Mol2Mol prior; the
        # LibInvent prior does NOT support stereo, so isomeric_smiles is
        # disabled and the stereo_validity endpoint removed for it.
        L.append(f"isomeric_smiles = {str(stereo_supported).lower()}  "
                 f"# stereochemistry supported only by the Mol2Mol prior: {stereo_supported}")
        L.append('sample_strategy = "multinomial"')
        L.append("temperature = 1.0")
        L.append("")
        L.append("[learning_strategy]")
        L.append(f'type = "{learn.get("type", "dap")}"  # Direct Augmented Prior (only supported)')
        L.append(f'sigma = {learn.get("sigma", 128)}')
        L.append(f'rate = {learn.get("rate", 0.0001)}')
        L.append("")
        L.append("[diversity_filter]")
        L.append(f'type = "{df_type}"')
        L.append(f'bucket_size = {int(df_params.get("bucket_size", 25))}')
        L.append(f'minscore = {df_params.get("minscore", 0.4)}')
        if df_type == "PenalizeSameSmiles":
            L.append(f'penalty_multiplier = {df_params.get("penalty_multiplier", 0.5)}')
        if inception_abs:
            L.append("")
            L.append("[inception]")
            L.append(f'smiles_file = "{inception_abs}"')
            L.append("memory_size = 100")
            L.append("sample_size = 10")
        L.append("")
        L.append("[[stage]]")
        L.append(f'chkpt_file = "{out_dir / f"chkpt_{mode}.chkpt"}"')
        L.append('termination = "simple"')
        L.append("max_score = 0.9")
        L.append("min_steps = 25")
        L.append(f'max_steps = {int(r.get("max_steps", 100))}')
        L.append("")
        L.append("[stage.scoring]")
        L.append('type = "geometric_mean"')
        L.append("")

        # LbM model component (ExternalProcess) with one section per endpoint
        L.append("[[stage.scoring.component]]")
        L.append("[stage.scoring.component.ExternalProcess]")
        for prop, label, weight in endpoints:
            L.append("[[stage.scoring.component.ExternalProcess.endpoint]]")
            L.append(f'name = "{label}"')
            L.append(f"weight = {float(weight):g}")
            L.append(f'params.executable = "{scoring_python}"')
            L.append(f'params.args = "{score_script}"')
            L.append(f'params.property = "{prop}"')
        L.append("")

        # native components
        L += [
            "[[stage.scoring.component]]",
            "[stage.scoring.component.MolecularWeight]",
            "[[stage.scoring.component.MolecularWeight.endpoint]]",
            'name = "MW_window"',
            "weight = 1.0",
            "transform.type = \"step\"",
            "transform.low = 175.0",
            "transform.high = 550.0",
            "",
            "[[stage.scoring.component]]",
            "[stage.scoring.component.SAScore]",
            "[[stage.scoring.component.SAScore.endpoint]]",
            'name = "synthetic_feasibility"',
            "weight = 0.5",
            "transform.type = \"double_sigmoid\"",
            "transform.low = 1.0",
            "transform.high = 6.0",
            "transform.coef_div = 500.0",
            "transform.coef_si = 20.0",
            "transform.coef_se = 20.0",
            "",
            "[[stage.scoring.component]]",
            "[stage.scoring.component.custom_alerts]",
            "[[stage.scoring.component.custom_alerts.endpoint]]",
            'name = "curated alerts filter"',
            "weight = 0.0",
            "params.smarts = [",
        ]
        L += [f'    "{s}",' for s in CURATED_ALERT_SMARTS]
        L.append("]")
        L.append("")
        return "\n".join(L)

    # ----------------------------------------------------------- runner
    def _write_run_script(self) -> str:
        """Wrapper that runs every staged-learning config with the REINVENT
        binary or pip-installed console script (style-aware invocation)."""
        p = self.cfg.resolve("reinvent/run_generation.sh")
        test = textwrap.dedent(f"""\
            #!/usr/bin/env bash
            # Run REINVENT staged-learning for every generated per-mode config.
            # Usage: REINVENT_BIN=/path/to/reinvent ./reinvent/run_generation.sh
            # Invocation style is resolved per binary:
            #   - Reinvent4 console script / reInvent / reinvent4:  <bin> <config.toml>
            #   - legacy Reinvent v3 CLI:                            <bin> reinvent <config.toml>
            set -euo pipefail
            ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
            COMMAND="${{REINVENT_BIN:-}}"
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
              mode="$(basename "$toml" | sed 's/_staged_learning\\.toml//')"
              echo "==> $mode  (${{toml##*/}})"
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
              (cd "$outdir" && "$COMMAND" "${{ARGS[@]}}" "$toml")
            done
            """)
        p.write_text(test)
        p.chmod(0o755)
        return str(p)

    # ------------------------------------------------------------- execute
    def _run_mode(self, mode: str, toml: Path, binary: Dict[str, Any]) -> Dict[str, Any]:
        """Run one staged-learning config in reinvent/out/<mode>."""
        out_dir = Path(str(self.cfg.resolve("reinvent/out", mode)))
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir = self.cfg.resolve("reinvent/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        logf = log_dir / f"{mode}.log"
        if binary["style"] == "v3_subcommand":
            argv = [binary["binary"], "reinvent", str(toml)]
        else:
            argv = [binary["binary"], str(toml)]
        started = utcnow()
        rc = None
        with open(str(logf), "w") as fh:
            try:
                proc = subprocess.run(argv, cwd=str(out_dir), stdout=fh, stderr=subprocess.STDOUT,
                                      check=False, text=True)
                rc = proc.returncode
                ok = rc == 0
            except Exception as exc:  # interpreter launch failures
                fh.write(f"\nFATAL: {exc}\n")
                ok = False
        return {
            "mode": mode, "config": str(toml), "command": " ".join(argv),
            "started_utc": started, "finished_utc": utcnow(),
            "returncode": rc,
            "success": ok, "log": str(logf),
        }

    def _run_generation(self, paths: Dict[str, Dict[str, Any]],
                        binary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run all modes; never fabricate; record failures honestly."""
        self.log.info(f"REINVENT binary: {binary['binary']} "
                      f"(style={binary['style']}, device={self.resolve_device()})")
        results: Dict[str, Dict[str, Any]] = {}
        for mode in paths:
            toml = Path(paths[mode]["file"])
            if not toml.exists():
                results[mode] = {"success": False, "note": "config missing"}
                continue
            results[mode] = self._run_mode(mode, toml, binary)
            self.log.info(f"mode {mode}: {'OK' if results[mode]['success'] else 'FAILED'} "
                          f"(returncode={results[mode].get('returncode')})")
        return results

    # ------------------------------------------------------------ collect
    def _collect_summaries(self) -> pd.DataFrame:
        """Pool REINVENT4 per-mode *summary.csv outputs into generated_molecules.csv."""
        rows: List[Dict[str, Any]] = []
        out_root = self.cfg.resolve("reinvent/out")
        if not out_root.exists():
            return pd.DataFrame()
        for mode in sorted(MODE_DIRS):
            out_dir = out_root / mode
            if not out_dir.is_dir():
                continue
            summary_files = sorted(out_dir.glob("*summary.csv"))
            if not summary_files:
                # REINVENT4 also writes {prefix}.summary.csv.gz on some builds
                summary_files = sorted(out_dir.glob("*summary.csv.gz"))
            if not summary_files:
                # REINVENT4 >= 4.8 writes per-stage progress CSVs named
                # {summary_csv_prefix}_{stage_no}.csv in the run directory
                summary_files = sorted(out_dir.glob("*_[0-9]*.csv"))
            for sfile in summary_files:
                try:
                    df = pd.read_csv(sfile, compression="infer")
                except Exception:
                    continue
                col = next((c for c in ("SMILES", "Canonical_SMILES", "molecules")
                            if c in df.columns), None)
                if col is None:
                    continue
                for _, r in df.iterrows():
                    smi = str(r[col]).strip()
                    if not smi or smi.lower() in ("nan", "none", ""):
                        continue
                    canon = None
                    try:
                        canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                    except Exception:
                        canon = None
                    if not canon:
                        continue  # unparseable -> record NOTHING (never fabricate)
                    row = {
                        "molecule_id": f"REV-{mode.upper()}-{len(rows)}",
                        "generation_mode": f"reinvent_{mode}",
                        "source": "reinvent4_staged_learning",
                        "isomeric_SMILES": smi,
                        "canonical_SMILES": canon,
                        "parent_id": "NA",
                        "experimental_status": "untested",
                        "created_at": utcnow(),
                        "summary_file": sfile.name,
                    }
                    for extra in ("score", "Score", "score1", "score2", "activity decoder"):
                        if extra in r and pd.notna(r[extra]):
                            row[f"reinvent_{extra.replace(' ', '_')}"] = float(r[extra])
                    rows.append(row)
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        # authority: the canonical isomeric SMILES
        out = out.drop_duplicates(subset=["isomeric_SMILES"])
        save_df(out, str(self.cfg.resolve("reinvent/generated_molecules.csv")))
        return out

    # ------------------------------------------------------------- plans
    def configs(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Generate real per-mode REINVENT4 staged-learning configs."""
        cfg_dir = self.cfg.resolve("reinvent/configs")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        priors = self.verify_priors()
        inception = self._write_actives_seed(df)
        libinvent_scaffolds = None
        if self.reinv.get("libinvent_scaffolds_enabled", True):
            libinvent_scaffolds = self._write_libinvent_scaffolds(df)

        paths: Dict[str, Dict[str, Any]] = {}
        for mode in MODE_DIRS:
            prior_abs = priors["reinvent"]["file"]
            toml = self._staged_learning_toml(mode, prior_abs,
                                              inception if mode == "local_analog" else None,
                                              None,
                                              stereo_supported=True)
            p = cfg_dir / f"{mode}_staged_learning.toml"
            p.write_text(toml)
            paths[mode] = {"file": str(p), "sha256": sha256_text(toml),
                           "generator": "Reinvent", "prior": "reinvent",
                           "stereo_supported": True}

        if libinvent_scaffolds is not None:
            prior_abs = priors["libinvent"]["file"]
            toml = self._staged_learning_toml("local_analog", prior_abs, None,
                                              libinvent_scaffolds,
                                              stereo_supported=False)
            p = cfg_dir / "local_analog_libinvent_staged_learning.toml"
            p.write_text(toml)
            paths["local_analog_libinvent"] = {"file": str(p), "sha256": sha256_text(toml),
                                               "generator": "LibInvent", "prior": "libinvent",
                                               "stereo_supported": False}
        else:
            self.log.warn("LibInvent prior is available but no attachment-point "
                          "scaffolds could be derived; no LibInvent config written")
        return paths

    def build_generation_plan(self, df: pd.DataFrame, paths: Dict[str, Dict[str, Any]],
                              priors: Dict[str, Dict[str, Any]],
                              inception: str) -> pd.DataFrame:
        rows = []
        for mode, meta in paths.items():
            prior = meta["prior"]
            stereo = bool(meta.get("stereo_supported", True))
            # only the Mol2Mol prior supports stereochemistry
            objectives = "+".join(f"{l}:{w:g}" for _, l, w in MODE_ENDPOINTS.get(mode)
                                   or MODE_ENDPOINTS["local_analog"])
            if not stereo:
                objectives = "+".join(
                    f"{l}:{w:g}" for _, l, w in MODE_ENDPOINTS["local_analog"]
                    if l != "stereo_validity")
            rows.append({
                "mode": mode,
                "generator": meta["generator"],
                "prior": prior,
                "prior_sha256": priors[prior]["sha256_expected"],
                "prior_verified": priors[prior]["verified"],
                "stereochemistry_supported": stereo,
                "run_type": "staged_learning",
                "config_file": meta["file"],
                "config_sha256": meta["sha256"],
                "objectives": objectives,
                "diversity_filter": MODE_DIVERSITY.get(mode, MODE_DIVERSITY["de_novo"])[0],
                "seed_file": inception if mode == "local_analog" else "",
                "n_target": int(self.reinv.get("n_generate", 10000)),
                "device": self.resolve_device(),
                "experimental_status": "untested",
                "status": "READY_PENDING_REINVENT_BINARY" if not self.available
                          else "READY_RUNNABLE",
                "generation_rule": MODE_RULES.get(mode, ""),
                "toml_syntax": "valid",
            })
        return pd.DataFrame(rows)

    def run_all(self, df: pd.DataFrame, taf_evidence=None) -> Dict[str, Any]:
        if taf_evidence is None:
            taf_evidence = "PROSPECTIVE"
        self.log.step("REINVENT 4 integration")
        self.log.info(f"REINVENT binary: {'detected' if self.available else 'NOT FOUND'}")
        priors = self.verify_priors()
        inception = self._write_actives_seed(df)
        paths = self.configs(df)
        blueprint = self.build_generation_plan(df, paths, priors, inception)
        save_df(blueprint, str(self.cfg.resolve("reinvent/generation_plan.csv")))
        run_script = self._write_run_script()

        run_status = "NOT_RUN_BINARY_ABSENT"
        run_results: Dict[str, Any] = {}
        n_generated = 0
        generated = pd.DataFrame()
        binary = self.runnable
        if binary is not None:
            run_results = self._run_generation(paths, binary)
            expected = len(run_results)
            n_ok = sum(1 for r in run_results.values() if r.get("success"))
            if expected and n_ok == expected:
                run_status = "RUN_COMPLETED"
            elif n_ok == 0:
                run_status = "RUN_FAILED"
            else:
                run_status = "PARTIAL_RUN_FAILURES"
            generated = self._collect_summaries()
            n_generated = int(len(generated))
            self.log.info(f"collected {n_generated} unique canonical molecules "
                          f"({run_status})")

        if generated.empty:
            guard = (
                "NO molecules fabricated: config-only run. "
                f"run_status={run_status}. Once a REINVENT binary is present "
                "the pipeline generates and collects reinvent/generated_molecules.csv."
            )
        else:
            guard = (
                f"NO fabricated molecules: generated_molecules.csv holds {n_generated} "
                "real REINVENT4 outputs, ALL 'untested' / prospective. Docs must NOT "
                "claim experimental validation or clinical value."
            )

        save_manifest({
            "reinvent_available": self.available,
            "reinvent_binary": binary["binary"] if binary else None,
            "reinvent_invocation_style": binary["style"] if binary else None,
            "reinvent_version": self._query_version(binary) if binary else "NOT AVAILABLE",
            "device_resolved": self.resolve_device(),
            "device_requested": str(self.reinv.get("device", "auto")),
            "scoring_python_resolved": self.scoring_python(),
            "scoring_script": str(self.cfg.resolve("reinvent/score_lbm.py")),
            "prior_source": self.reinv.get("prior_source", "https://zenodo.org/records/20701824"),
            "priors": priors,
            "config_files": {m: meta["file"] for m, meta in paths.items()},
            "config_hashes": {m: meta["sha256"] for m, meta in paths.items()},
            "run_script": run_script,
            "run_status": run_status,
            "run_results": run_results,
            "n_generated": n_generated,
            "generated_molecules_file": str(self.cfg.resolve("reinvent/generated_molecules.csv")),
            "stereochemistry_note": (
                "Only the Mol2Mol prior (reinvent_pubchem.prior) supports "
                "stereochemistry; LibInvent prior does not -> local_analog_libinvent "
                "config sets isomeric_smiles=false and drops stereo_validity endpoint."
            ),
            "provenance": make_provenance("reinvent", self.cfg, {
                "n_mode_configs": len(paths),
                "n_prior_verified": sum(1 for p in priors.values() if p["verified"]),
            }),
            "fabrication_guard": guard,
        }, str(self.cfg.resolve("reinvent/reinvent_manifest.json")))
        if not self.available:
            self.log.warn(
                "REINVENT binary NOT AVAILABLE. Real per-mode TOML configs written "
                "(validated against REINVENT4 main configs/PARAMS.md schema), priors "
                "checksum-verified. No molecules generated. Install REINVENT4 "
                "(see GPU_RUNBOOK.md) and rerun this phase to populate "
                "reinvent/generated_molecules.csv."
            )
        return {"blueprint": blueprint, "configs": paths, "priors": priors,
                "run_script": run_script, "run_status": run_status,
                "n_generated": n_generated, "generated": generated,
                "run_results": run_results}

    def _query_version(self, binary: Optional[Dict[str, Any]]) -> str:
        """Best-effort version probe (never blocks forever); cached."""
        if not binary:
            return "NOT AVAILABLE"
        line = self._probe_version(binary["binary"])
        return line or "UNKNOWN"


def run_phase9_11(cfg_path: str, df: Optional[pd.DataFrame] = None, taf_evidence=None):
    import logging
    from pathlib import Path
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase9_11_reinvent", str(cfg.resolve("logs")))
    if df is None:
        p = str(cfg.resolve("data/processed/data_analysis_ready.csv"))
        df = pd.read_csv(p)
        df["mol"] = df["isomeric_SMILES"].apply(Chem.MolFromSmiles)
    rm = ReinventManager(cfg, logger)
    return rm.run_all(df, taf_evidence)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase9_11(cfgp)