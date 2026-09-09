#!/usr/bin/env python3
"""Phase 13: Candidate prioritization and portfolio construction.

Implements prompt2.txt sections 27, 21(prospective), 24 (provenance) and
report 9:
  - candidate scoring across TAF / novelty / diversity / uncertainty / AD axes
  - never select only top predicted actives
  - portfolio groups A-F (exploitation, falsification, transfer, novelty,
    information gain, uncertainty reduction)
  - full generation provenance record (experimental_status=untested)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest, utcnow,
)

RDLogger.DisableLog("rdApp.*")

DESC_COLS = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
             "RingCount", "AromaticRings", "FractionCSP3", "HeavyAtomCount"]


def morgan(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _load_frozen(cfg: ProjectConfig) -> Optional[Dict[str, Any]]:
    """Load the frozen Phase-4 classifier artifact (used by REINVENT scoring)."""
    import joblib
    p = cfg.resolve("models/classical/lbm_rf_fp_desc.joblib")
    if not p.exists():
        return None
    return joblib.load(p)


class CandidatePrioritizer:
    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger

    def score_candidates(self, df: pd.DataFrame, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        self.log.step("Candidate scoring (multi-axis)")
        train_fps = [morgan(m) for m in df["mol"].tolist() if m is not None]
        act_fps = [morgan(m) for m in df[df["is_active"] == 1]["mol"] if m is not None]
        actives = df[df["is_active"] == 1]

        frozen = _load_frozen(self.cfg)
        clf = frozen["model"] if frozen else None
        scaler = frozen["scaler"] if frozen else None
        art_desc_cols = [c for c in (frozen or {}).get("desc_columns", DESC_COLS) if c in DESC_COLS]

        def _featurize(mol):
            from src.models import morgan_array
            fp = morgan_array(mol)
            d = {
                "MW": Descriptors.MolWt(mol), "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol), "HBD": Descriptors.NumHDonors(mol),
                "HBA": Descriptors.NumHAcceptors(mol), "RotBonds": Descriptors.NumRotatableBonds(mol),
                "RingCount": Descriptors.RingCount(mol),
                "AromaticRings": Descriptors.NumAromaticRings(mol),
                "FractionCSP3": Descriptors.FractionCSP3(mol),
                "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
            }
            return np.hstack([fp, [d[c] for c in art_desc_cols]]).astype(np.float64)

        out = []
        for r in rows:
            mol = Chem.MolFromSmiles(r.get("isomeric_SMILES") or r.get("smiles"))
            if mol is None:
                continue
            fp = morgan(mol)
            sim_train = max(DataStructs.TanimotoSimilarity(fp, t) for t in train_fps)
            sim_active = max(DataStructs.TanimotoSimilarity(fp, t) for t in act_fps) if act_fps else 0

            # TAF proxy scoring from explicit feature checks (2D only here)
            taf_score = 0.0
            for sm in ["C(=O)O[C,C]", "[CX4][OH]", "[Br]", "C#C"]:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(sm)):
                    taf_score += 1.0 / 4.0

            # predicted activity + predictive uncertainty from the frozen model (section 10/11)
            predicted_activity = np.nan
            predictive_uncertainty = np.nan
            if clf is not None and scaler is not None:
                try:
                    Xrow = _featurize(mol).reshape(1, -1)
                    Xs = scaler.transform(Xrow)
                    proba = clf.predict_proba(Xs)[0, 1]
                    predicted_activity = round(float(proba), 4)
                    if hasattr(clf, "estimators_") and clf.estimators_ is not None:
                        tree_ps = [t.predict_proba(Xs)[0, 1] for t in clf.estimators_]
                        predictive_uncertainty = round(float(np.std(tree_ps)), 4)
                    else:
                        predictive_uncertainty = round(float(1.0 - abs(proba - 0.5) * 2.0), 4)
                except Exception:
                    pass

            uncertainty = r.get("uncertainty", np.nan) if np.isfinite(r.get("uncertainty", np.nan)) \
                else (predictive_uncertainty if np.isfinite(predictive_uncertainty) else 0.5)

            # information-gain acquisition function (section 36): candidates near the
            # decision boundary (|p-0.5| small) maximally discriminate competing
            # activity hypotheses H1-H6 (uncertainty sampling on the frozen model)
            info_gain = (0.5 - abs(predicted_activity - 0.5)) if np.isfinite(predicted_activity) else np.nan
            if not np.isfinite(info_gain):
                info_gain = predictive_uncertainty if np.isfinite(predictive_uncertainty) else 0.5

            # 3-state applicability domain mirroring validation/applicability_domain.py (section 10)
            if sim_active and sim_active > 0:
                d_active = 1 - sim_active
            else:
                d_active = np.nan
            d_any = 1 - sim_train
            if d_any < 0.35:
                ad = "IN-DOMAIN"
            elif d_any < 0.6 or (np.isfinite(d_active) and d_active < 0.45):
                ad = "NEAR-DOMAIN"
            else:
                ad = "OUT-OF-DOMAIN"

            out.append({
                "molecule_id": r.get("molecule_id", "UNSET"),
                "generation_mode": r.get("generation_mode", "unknown"),
                "source": r.get("source", "unknown"),
                "SMILES": r.get("isomeric_SMILES") or r.get("smiles"),
                "isomeric_SMILES": r.get("isomeric_SMILES") or r.get("smiles"),
                "parent_id": r.get("parent_id", np.nan),
                "TAF_score_2d": round(taf_score, 4),
                "predicted_activity": predicted_activity,
                "predictive_uncertainty": predictive_uncertainty,
                "similarity_to_train_max": round(float(sim_train), 4),
                "similarity_to_active_max": round(float(sim_active), 4),
                "novelty": round(float(1 - sim_train), 4),
                "uncertainty": round(float(uncertainty), 4),
                "information_gain_score": round(float(info_gain), 4),
                "applicability_domain": ad,
                "selection_reason": r.get("selection_reason", ""),
                "experimental_status": "untested",
                "created_at": utcnow(),
            })
        return pd.DataFrame(out)

    def add_controls(self, df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        """Construct TAF-disrupting negative controls from active parents (section 27B/21D)."""
        rows = []
        for _, r in df.iterrows():
            if r["is_active"] != 1:
                continue
            mol = r["mol"]
            if mol is None:
                continue
            # remove OH (TAF-2 knock-out)
            for name, sm in [("no_OH", "[CX4][OH]"), ("no_Br_or_alkyne", "[Br]")]:
                patt = Chem.MolFromSmarts(sm)
                if mol.HasSubstructMatch(patt):
                    repl = Chem.MolFromSmiles("C")
                    try:
                        n = Chem.ReplaceSubstructs(mol, patt, repl, replaceAll=True)
                        for m in n:
                            smi = Chem.MolToSmiles(m, isomericSmiles=True)
                            rows.append({
                                "molecule_id": f"NEG-{int(r['Identifier'])}-{name}",
                                "generation_mode": "negative_control",
                                "source": "control_design",
                                "isomeric_SMILES": smi,
                                "parent_id": int(r["Identifier"]),
                                "selection_reason": f"TAF-disrupting control: {name}",
                                "experimental_status": "untested",
                                "uncertainty": 0.6,
                            })
                    except Exception:
                        continue
        ctrl = self.score_candidates(df, rows)
        if len(candidates):
            out = pd.concat([candidates, ctrl], ignore_index=True)
        else:
            out = ctrl
        return out

    def portfolio(self, candidates: pd.DataFrame) -> pd.DataFrame:
        self.log.step("Candidate portfolio (groups A-F)")
        if not len(candidates):
            return candidates
        buckets = []
        nov = candidates["novelty"]
        # Group A: high confidence TAF preserving (high TAF, near domain)
        a = candidates[(candidates["TAF_score_2d"] >= 0.75) & (candidates["applicability_domain"] == "NEAR-DOMAIN")]
        for _, r in a.iterrows():
            buckets.append({**r.to_dict(), "portfolio_group": "A_high_confidence_taf_preserving"})
        # Group B: TAF-disrupting (negative controls)
        b = candidates[candidates["generation_mode"] == "negative_control"]
        for _, r in b.iterrows():
            buckets.append({**r.to_dict(), "portfolio_group": "B_taf_disrupting"})
        # Group C: scaffold transfer (novelty high, moderate TAF)
        c = candidates[(nov > 0.5) & (candidates["TAF_score_2d"] >= 0.5) & (candidates["generation_mode"] != "negative_control")]
        for _, r in c.iterrows():
            buckets.append({**r.to_dict(), "portfolio_group": "C_scaffold_transfer"})
        # Group D: chemically novel (high novelty regardless of TAF)
        d = candidates[nov > nov.quantile(0.7)]
        for _, r in d.iterrows():
            buckets.append({**r.to_dict(), "portfolio_group": "D_chemically_novel"})
        # Group E: information gain (near decision boundary -> discriminates H1-H6)
        e = candidates[candidates["information_gain_score"] >= 0.3]
        for _, r in e.iterrows():
            bucket = {**r.to_dict(), "portfolio_group": "E_information_gain"}
            bucket["hypothesis_linked"] = ("Candidates near the frozen-model decision boundary "
                                           "maximally discriminate competing hypotheses H1-H6 "
                                           "(section 36).")
            buckets.append(bucket)
        # Group F: uncertainty reduction (OOD)
        f = candidates[candidates["applicability_domain"] == "OUT-OF-DOMAIN"]
        for _, r in f.iterrows():
            buckets.append({**r.to_dict(), "portfolio_group": "F_uncertainty_reduction"})

        if buckets:
            port = pd.DataFrame(buckets).drop_duplicates(subset=["molecule_id", "portfolio_group"], keep="first")
        else:
            port = candidates.copy()
            port["portfolio_group"] = "A_high_confidence_taf_preserving"
        save_df(port, str(self.cfg.resolve("prospective/prospective_candidates.csv")))
        self.log.info(f"Portfolio size: {len(port)}; groups reflexive overlap expected")
        return port

    def run_all(self, df: pd.DataFrame, generated: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.log.step("Candidate prioritization")
        rows: List[Dict[str, Any]] = []
        if generated is not None and len(generated):
            for _, r in generated.iterrows():
                rows.append({
                    "molecule_id": r.get("molecule_id", "GEN-UNSET"),
                    "generation_mode": r.get("generation_mode", "generated"),
                    "source": r.get("source", "generated"),
                    "isomeric_SMILES": r.get("isomeric_SMILES") or r.get("generated_SMILES"),
                    "parent_id": r.get("parent_id", np.nan),
                    "selection_reason": r.get("selection_reason", ""),
                    "uncertainty": 0.5,
                })
        cand = self.score_candidates(df, rows)
        cand = self.add_controls(df, cand)
        port = self.portfolio(cand)
        save_manifest(
            make_provenance("candidates", self.cfg, {"n_candidates": int(len(cand))}),
            str(self.cfg.resolve("prospective/candidate_manifest.json")),
        )
        return port


def run_phase13(cfg_path: str, df: Optional[pd.DataFrame] = None,
                generated: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    import os, logging
    from pathlib import Path
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase13_candidates", str(cfg.resolve("logs")))
    if df is None:
        p = str(cfg.resolve("data/processed/data_analysis_ready.csv"))
        df = pd.read_csv(p)
        df["mol"] = df["isomeric_SMILES"].apply(Chem.MolFromSmiles)
    if generated is None:
        g = str(cfg.resolve("generated/medchem_baseline_analogs.csv"))
        generated = pd.read_csv(g) if os.path.exists(g) else pd.DataFrame()
    prio = CandidatePrioritizer(cfg, logger)
    return prio.run_all(df, generated=generated)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase13(cfgp)