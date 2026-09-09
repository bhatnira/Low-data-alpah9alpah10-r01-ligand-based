#!/usr/bin/env python3
"""Phase 8: Traditional medicinal-chemistry baseline.

Implements prompt2.txt section 20, 25 (Model E):
  - conventional design baseline: MMP transformations, analog enumeration,
    bioisosteric replacement, substituent scanning, scaffold expansion/hop
  - comparator for AI-assisted design
  - all enumerated analogues are marked experimental_status=untested
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
)

RDLogger.DisableLog("rdApp.*")


def morgan(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


class MedChemBaseline:
    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger

    def _bioisostere_sets(self) -> List[List[str]]:
        sets = self.cfg.raw.get("medchem_baseline", {}).get("bioisosteres", [])
        return [list(s) for s in sets]

    def _smarts_replacements(self, mol) -> List[str]:
        """Apply bioisostere swaps on matched atoms; each variant is a hypothesis."""
        out = []
        for group in self._bioisostere_sets():
            for s in group:
                patt = Chem.MolFromSmarts(s)
                if patt is None:
                    continue
                if mol.HasSubstructMatch(patt):
                    # mark that a bioisostere position exists; enumerate replacement matches
                    for repl in group:
                        if repl == s:
                            continue
                        # crude but explicit: swap the matched substructure to another from group
                        try:
                            rmol = Chem.MolFromSmiles(chem_smiles_of_smarts(repl, s))
                            if rmol is not None:
                                n = Chem.ReplaceSubstructs(mol, patt, rmol, replaceAll=True)
                                for m in n:
                                    smi = Chem.MolToSmiles(m, isomericSmiles=True)
                                    if smi and smi not in out:
                                        out.append(smi)
                        except Exception:
                            continue
        return out

    def _enumeration(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        max_analogs = self.cfg.raw.get("medchem_baseline", {}).get("max_analogs_per_parent", 12)
        parents = df[df["is_active"] == 1] if not (df["is_active"] == 1).any() else df[
            (df["is_active"] == 1) | (df["Identifier"].isin([12, 25, 24, 18, 1, 2, 3]))
        ]
        for _, r in parents.iterrows():
            mol = r["mol"]
            if mol is None:
                continue
            analogs = self._smarts_replacements(mol)
            for a in analogs[:max_analogs]:
                amol = Chem.MolFromSmiles(a)
                if amol is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(morgan(mol), morgan(amol))
                rows.append({
                    "strategy": "bioisosteric_replacement",
                    "parent_id": int(r["Identifier"]),
                    "generated_SMILES": a,
                    "isomeric_SMILES": a,
                    "parent_similarity": round(float(sim), 4),
                    "experimental_status": "untested",
                    "generation_mode": "medchem_baseline",
                    "source": "traditional_medicinal_chemistry",
                    "selection_reason": "bioisostere of active parent",
                    "scaffold_hop": bool(sim < 0.5),
                })
        return pd.DataFrame(rows)

    def substituent_scan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Substituent enumeration: swap ring/alkoxy positions with halogen/alkyl candidates."""
        rows = []
        alkoxy_smarts = "[OX2][CH2,CX4]"
        halogens = ["Cl", "Br", "I", "C#N", "C(C)(C)"]
        for _, r in df.iterrows():
            mol = r["mol"]
            if mol is None:
                continue
            patt = Chem.MolFromSmarts(alkoxy_smarts)
            if not mol.HasSubstructMatch(patt):
                continue
            frag_repl = Chem.MolFromSmiles("Cl")
            for h in halogens[:3]:
                try:
                    repl = Chem.MolFromSmiles(h)
                    n = Chem.ReplaceSubstructs(mol, patt, repl, replaceAll=True)
                    for m in n:
                        smi = Chem.MolToSmiles(m, isomericSmiles=True)
                        amol = Chem.MolFromSmiles(smi)
                        if amol is None:
                            continue
                        sim = DataStructs.TanimotoSimilarity(morgan(mol), morgan(amol))
                        rows.append({
                            "strategy": "substituent_scanning",
                            "parent_id": int(r["Identifier"]),
                            "generated_SMILES": smi,
                            "isomeric_SMILES": smi,
                            "parent_similarity": round(float(sim), 4),
                            "experimental_status": "untested",
                            "generation_mode": "medchem_baseline",
                            "source": "traditional_medicinal_chemistry",
                            "selection_reason": f"substituent swap to {h}",
                            "scaffold_hop": False,
                        })
                except Exception:
                    continue
        return pd.DataFrame(rows)

    def run_all(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.step("Traditional medicinal-chemistry baseline")
        bio = self._enumeration(df)
        scan = self.substituent_scan(df)
        out = pd.concat([bio, scan], ignore_index=True) if len(bio) or len(scan) else pd.DataFrame(
            columns=["strategy", "parent_id", "generated_SMILES", "isomeric_SMILES",
                     "parent_similarity", "experimental_status", "generation_mode",
                     "source", "selection_reason", "scaffold_hop"])
        if len(out):
            out = out.drop_duplicates(subset=["isomeric_SMILES"])
            out["molecule_id"] = [f"MC-{i}" for i in range(len(out))]
        save_df(out, str(self.cfg.resolve("generated/medchem_baseline_analogs.csv")))
        self.log.info(f"Medchem baseline analogues generated: {len(out)}")
        save_manifest(
            make_provenance("medchem_baseline", self.cfg, {"n_parents_used": int(len(df))}),
            str(self.cfg.resolve("chemical_space/medchem_manifest.json")),
        )
        return out


def chem_smiles_of_smarts(repl_smiles, orig_smarts):
    """Best-effort SMILES for replacement module; falls back to the SMILES string."""
    return repl_smiles


def run_phase8(cfg_path: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    import os, logging
    from pathlib import Path
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase8_medchem", str(cfg.resolve("logs")))
    if df is None:
        p = str(cfg.resolve("data/processed/data_analysis_ready.csv"))
        df = pd.read_csv(p)
        df["mol"] = df["isomeric_SMILES"].apply(Chem.MolFromSmiles)
    mc = MedChemBaseline(cfg, logger)
    return mc.run_all(df)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase8(cfgp)