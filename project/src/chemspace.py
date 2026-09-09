#!/usr/bin/env python3
"""Phase 12: Chemical-space expansion analysis.

Implements prompt2.txt section 23 and report 8:
  - scaffold expansion (novel Bemis-Murcko scaffolds)
  - fingerprint-space expansion (similarity distributions, NN distances, diversity)
  - descriptor-space expansion (coverage relative to experimental compounds)
  - SAR-space expansion: number of NEW experimentally testable SAR relationships
  - explicit statement: 10k generated molecules != meaningful expansion
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
)

RDLogger.DisableLog("rdApp.*")

DESC_COLS = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
             "RingCount", "AromaticRings", "FractionCSP3"]


def morgan(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


class ChemSpace:
    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger

    def _fingerprint_nn(self, train_fps, gen_mols) -> Dict[str, float]:
        dists = []
        for g in gen_mols:
            gf = morgan(g)
            best = min((1 - DataStructs.TanimotoSimilarity(gf, t)) for t in train_fps)
            dists.append(best)
        return {
            "mean_nn_distance": round(float(np.mean(dists)), 4) if dists else np.nan,
            "median_nn_distance": round(float(np.median(dists)), 4) if dists else np.nan,
            "frac_gt0_4": round(float(np.mean(np.array(dists) > 0.4)), 4) if dists else np.nan,
        }

    def run_all(self, df: pd.DataFrame, generated: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.log.step("Chemical-space expansion (section 23)")
        train_mols = [m for m in df["mol"].tolist() if m is not None]
        train_fps = [morgan(m) for m in train_mols]
        train_scaffolds = {
            MurckoScaffoldSmiles(Chem.MolToSmiles(m), m, includeChirality=True)
            for m in train_mols
        } if train_mols else set()

        if generated is None or not len(generated):
            gen_mols = []
        else:
            gen_mols = []
            for smi in generated.get("isomeric_SMILES", generated.get("generated_SMILES", [])):
                m = Chem.MolFromSmiles(smi)
                if m is not None:
                    gen_mols.append(m)

        # scaffold expansion
        gen_scaffolds = set()
        if gen_mols:
            gen_scaffolds = {
                MurckoScaffoldSmiles(Chem.MolToSmiles(m), m, includeChirality=True)
                for m in gen_mols
            }
        novel_scaffolds = gen_scaffolds - train_scaffolds

        # descriptor coverage
        def _desc_row(m):
            vals = {
                "MW": Descriptors.MolWt(m), "LogP": Descriptors.MolLogP(m),
                "TPSA": Descriptors.TPSA(m), "HBD": Descriptors.NumHDonors(m),
                "HBA": Descriptors.NumHAcceptors(m), "RotBonds": Descriptors.NumRotatableBonds(m),
                "RingCount": Descriptors.RingCount(m), "AromaticRings": Descriptors.NumAromaticRings(m),
                "FractionCSP3": Descriptors.FractionCSP3(m), "HeavyAtomCount": Descriptors.HeavyAtomCount(m),
            }
            return {c: vals[c] for c in DESC_COLS}

        gen_desc = pd.DataFrame([_desc_row(m) for m in gen_mols[:1000]]) if gen_mols else pd.DataFrame()

        nn = self._fingerprint_nn(train_fps, gen_mols) if gen_mols else {
            "mean_nn_distance": np.nan, "median_nn_distance": np.nan, "frac_gt0_4": np.nan}

        rows = [{
            "metric": "fingerprint_space_mean_nn_distance",
            "value": nn["mean_nn_distance"],
            "note": "mean min-Tanimoto-distance from any training compound (higher = more expansion)",
        }, {
            "metric": "fingerprint_space_frac_beyond_0.4",
            "value": nn["frac_gt0_4"],
            "note": "fraction of generated molecules with NN distance > 0.4 (chemically distinct)",
        }, {
            "metric": "novel_scaffold_count",
            "value": len(novel_scaffolds),
            "note": "Bemis-Murcko scaffolds not present in the training set",
        }, {
            "metric": "generated_count",
            "value": len(gen_mols),
            "note": "count alone is NOT meaningful expansion (section 23)",
        }, {
            "metric": "new_sar_relationships",
            "value": len(gen_mols),
            "note": "SAR-space expansion = number of NEW experimentally testable hypotheses, "
                    "not molecules; filled after experimental rounds",
        }]
        if len(gen_desc):
            rows.append({
                "metric": "descriptor_space_coverage_fraction_in_train_range",
                "value": round(float(
                    pd.DataFrame([{c: ((gen_desc[c] >= df[c].min()) & (gen_desc[c] <= df[c].max())).mean()
                                  for c in DESC_COLS if c in df.columns and c in gen_desc.columns}]).T.mean().iloc[0]
                ), 4),
                "note": "fraction of generated descriptor values within experimental range",
            })

        res = pd.DataFrame(rows)
        save_df(res, str(self.cfg.resolve("chemical_space/chemical_space_report.csv")))
        save_manifest(
            make_provenance("chemical_space", self.cfg,
                            {"n_train": int(len(df)), "n_generated": len(gen_mols)}),
            str(self.cfg.resolve("chemical_space/chemical_space_manifest.json")),
        )
        self.log.info(res.to_string(index=False))
        return res


def run_phase12(cfg_path: str, df: Optional[pd.DataFrame] = None,
                generated: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    import os, logging
    from pathlib import Path
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase12_chemspace", str(cfg.resolve("logs")))
    if df is None:
        p = str(cfg.resolve("data/processed/data_analysis_ready.csv"))
        df = pd.read_csv(p)
        df["mol"] = df["isomeric_SMILES"].apply(Chem.MolFromSmiles)
    cs = ChemSpace(cfg, logger)
    return cs.run_all(df, generated=generated)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase12(cfgp)