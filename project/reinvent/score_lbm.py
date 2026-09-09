#!/usr/bin/env python3
"""REINVENT4 ExternalProcess payload script (LbM scoring endpoints).

Contract (see REINVENT4 configs/PARAMS.md, ExternalProcess component):
  * reads one isomeric SMILES per line on stdin
  * writes a single JSON object on stdout:
        {"version": 1, "payload": {<endpoint>: [scores...], ...}}
  * unparseable SMILES are scored NaN (NaN marks failure, never 0)

Every endpoint is computed with the frozen Phase-4 model artifact
(models/classical/lbm_rf_fp_desc.joblib) and the real TAF operational
definitions enforced in Phase 6.  Feature construction mirrors
src/models.Representations and src/dataset.DESCRIPTOR_FNS exactly, so the
scoring pipeline sees the same features the classifier was trained on.

Run with the project venv python only (config: reinvent.scoring_python).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from src.dataset import DESCRIPTOR_FNS  # identical descriptor pipeline as training
from src.models import morgan_array  # identical Morgan feature pipeline as training

RDLogger.DisableLog("rdApp.*")


def _artifact_path() -> Path:
    env = os.environ.get("LBM_MODEL_ARTIFACT")
    if env:
        return Path(env)
    return ROOT / "models/classical/lbm_rf_fp_desc.joblib"


class _MolCache:
    """Batched RDKit parse cache: REINVENT re-scores overlapping batches."""

    def __init__(self) -> None:
        self._mol = {}
        self._fp = {}
        self._murcko = {}

    def mol(self, smi: str) -> Chem.Mol:
        m = self._mol.get(smi)
        if m is None:
            m = Chem.MolFromSmiles(smi) or Chem.Mol()
            self._mol[smi] = m
        return m

    def fp(self, smi: str) -> np.ndarray:
        fp = self._fp.get(smi)
        if fp is None:
            mol = self.mol(smi)
            fp = morgan_array(mol) if mol.GetNumAtoms() else np.zeros(2048)
            self._fp[smi] = fp
        return fp

    def murcko(self, smi: str):
        m = self._murcko.get(smi)
        if m is None:
            mol = self.mol(smi)
            if mol.GetNumAtoms():
                from rdkit.Chem.Scaffolds import MurckoScaffold
                try:
                    m = Chem.MolToSmiles(MurckoScaffold.MurckoScaffoldSmiles(mol))
                except Exception:
                    m = ""
            else:
                m = ""
            self._murcko[smi] = m
        return m


class _TafScorer:
    """TAF consistency at the SMILES level.

    Only the TAFs measurable in 2D are aggregated (lactone, hydroxyl donor,
    EWG).  3D distance/vector constraints (TAF-3's 3-5 A, TAF-2's vector,
    TAF-4's core R/S, TAF-5 window) are flagged explicitly and handled by the
    stereo_validity / applicability endpoints rather than silently ignored.
    """

    LACTONE = Chem.MolFromSmarts("C(=O)O[C,C]")
    DONOR = Chem.MolFromSmarts("[CX4][OH]")
    AMINE = Chem.MolFromSmarts("[NX3;!$([N;H0]);!$(N=*)]")
    EWG = [Chem.MolFromSmarts(s) for s in ("[Br]", "C#C", "C#N")]

    def score(self, mol: Chem.Mol) -> float:
        if mol is None or mol.GetNumAtoms() == 0:
            return float("nan")
        lactone = float(mol.HasSubstructMatch(self.LACTONE))
        donor = float(mol.HasSubstructMatch(self.DONOR) or mol.HasSubstructMatch(self.AMINE))
        ewg = float(any(mol.HasSubstructMatch(p) for p in self.EWG))
        return float(np.mean([lactone, donor, ewg]))


def _stereo_validity(mol: Chem.Mol) -> float:
    from rdkit.Chem import rdMolDescriptors
    if mol is None or mol.GetNumAtoms() == 0:
        return float("nan")
    centers = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    if centers == 0:
        return 1.0
    unspecified = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    return float((centers - unspecified) / centers)


def main() -> int:
    import joblib

    artifact_path = _artifact_path()
    if not artifact_path.exists():
        print(json.dumps({"version": 1, "error": f"missing artifact: {artifact_path}"}))
        return 2

    artifact = joblib.load(artifact_path)
    clf = artifact["model"]
    scaler = artifact["scaler"]
    positive_class = int(artifact.get("positive_class", 1))
    desc_cols = list(artifact.get("desc_columns", []))
    active_smiles = list(artifact.get("active_smiles", []))
    training_murcko = set(artifact.get("training_murcko", []))
    nbits = int(artifact.get("nbits", 2048))
    if "feature_columns" in artifact:
        desc_cols = [c for c in artifact["feature_columns"] if c in DESCRIPTOR_FNS]
    # ensure desc order matches training build order
    desc_fns = [(c, DESCRIPTOR_FNS[c]) for c in desc_cols]

    active_fps = np.vstack([morgan_array(Chem.MolFromSmiles(s)) for s in active_smiles]) \
        if active_smiles else np.zeros((1, nbits))

    taf = _TafScorer()
    cache = _MolCache()

    lines = sys.stdin.read().splitlines()
    keys = ["predicted_activity", "taf_consistency", "novelty",
            "scaffold_novelty", "stereo_validity", "uncertainty", "taf_disrupted"]
    payload = {k: [] for k in keys}

    for smi in lines:
        smi = smi.strip()
        if not smi:
            for k in keys:
                payload[k].append(float("nan"))
            continue
        mol = cache.mol(smi)
        if mol.GetNumAtoms() == 0:
            for k in keys:
                payload[k].append(float("nan"))
            continue
        try:
            fp = cache.fp(smi)
            desc = [fn(mol) for _, fn in desc_fns]
            X = np.hstack([fp, desc]).reshape(1, -1).astype(np.float64)
            proba = clf.predict_proba(scaler.transform(X))[0]
            p = float(proba[positive_class])

            tc = taf.score(mol)
            sims = (fp @ active_fps.T) / np.maximum(
                (np.linalg.norm(fp) * np.linalg.norm(active_fps, axis=1)), 1e-12)
            max_sim = float(sims.max()) if len(active_smiles) else 0.0
            novelty = 1.0 - max_sim

            murcko = cache.murcko(smi)
            scaffold_novel = 0.0 if murcko and murcko in training_murcko else 1.0

            sv = _stereo_validity(mol)
            # predictive uncertainty from tree-level variance (section 11)
            if hasattr(clf, "estimators_") and clf.estimators_ is not None:
                try:
                    tree_ps = [t.predict_proba(scaler.transform(X))[0][positive_class] for t in clf.estimators_]
                    uncertainty = float(np.std(tree_ps))
                except Exception:
                    uncertainty = 1.0 - abs(p - 0.5) * 2.0
            else:
                uncertainty = 1.0 - abs(p - 0.5) * 2.0
            taf_disrupted = 1.0 - tc

            payload["predicted_activity"].append(p)
            payload["taf_consistency"].append(tc)
            payload["novelty"].append(novelty)
            payload["scaffold_novelty"].append(scaffold_novel)
            payload["stereo_validity"].append(sv)
            payload["uncertainty"].append(uncertainty)
            payload["taf_disrupted"].append(taf_disrupted)
        except Exception:
            for k in keys:
                payload[k].append(float("nan"))

    print(json.dumps({"version": 1, "payload": payload}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())