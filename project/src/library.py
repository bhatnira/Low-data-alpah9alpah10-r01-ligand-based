#!/usr/bin/env python3
"""Phase 19: Chemotype-library projection and coverage pre-registration.

Deliverable upgrade for the R01: convert the candidate portfolio into an explicit
LIBRARY OF CHEMOTYPES - named chemotype families, round-by-round coverage
targets, scaffold capping rules, and honest external-validation status.

Honesty contract (unchanged): no fabricated molecules, no fabricated external
data. External/SAR-enrichment metrics are reported as computed or NOT AVAILABLE.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
)

RDLogger.DisableLog("rdApp.*")

TIER_RULE = (
    "tier1 = novel Bemis-Murcko scaffold vs training; "
    "tier2 = known scaffold in hypothesis groups A/B/C/E; "
    "tier3 = residual uncertainty-reduction set (mostly F)"
)


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Largest fragment by heavy-atom count (disconnects salts/component mixtures)."""
    if mol is None:
        return None
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    if not frags:
        return mol
    heaviest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(heaviest)
    except Exception:
        pass
    return heaviest


def _scaffold_smiles(mol: Chem.Mol) -> str:
    """Chemotype key: Bemis-Murcko scaffold, else largest-fragment SMILES."""
    single = _largest_fragment(mol)
    if single is None:
        return ""
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(single)
        s = Chem.MolToSmiles(scaff)
        if s:
            return s
    except Exception:
        pass
    return Chem.MolToSmiles(single, canonical=True)


def _arom_rings(mol: Chem.Mol) -> int:
    single = _largest_fragment(mol)
    return int(rdMolDescriptors.CalcNumAromaticRings(single)) if single is not None else 0


def _druglike_flags(mol: Chem.Mol) -> Dict[str, Any]:
    """Lipinski violations + PAINS + BRENK reactivity flags (None -> all unknown)."""
    mol = _largest_fragment(mol)
    if mol is None:
        return {"lipinski_violations": np.nan, "pains_flag": np.nan,
                "brenk_flag": np.nan, "druglike_lipinski": np.nan,
                "pains_clean": np.nan, "brenk_clean": np.nan}
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    viol = int((mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10))
    pains = bool(PAINS_CAT.HasMatch(mol))
    brenk = bool(BRENK_CAT.HasMatch(mol))
    return {"lipinski_violations": viol,
            "pains_flag": int(pains),
            "brenk_flag": int(brenk),
            "druglike_lipinski": int(viol <= 2),
            "pains_clean": int(not pains),
            "brenk_clean": int(not brenk)}


def _build_pains_catalog():
    p = FilterCatalog.FilterCatalogParams()
    p.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(p)


def _build_brenk_catalog():
    p = FilterCatalog.FilterCatalogParams()
    p.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(p)


PAINS_CAT = _build_pains_catalog()
BRENK_CAT = _build_brenk_catalog()


class ChemotypeLibrary:
    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger

    def project(self, portfolio: pd.DataFrame,
                training: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        self.log.step("Chemotype library projection (R01 deliverable, phase 19)")
        if not len(portfolio):
            self.log.warn("Empty portfolio; library projection skipped")
            return {}
        pf = portfolio.copy()
        pf["mol"] = pf["isomeric_SMILES"].apply(Chem.MolFromSmiles)
        pf["chemotype_scaffold"] = pf["mol"].apply(_scaffold_smiles)
        # novelty vs the 30-compound training library (has murcko_scaffold column)
        train_scaffolds = set(training["murcko_scaffold"].astype(str)) if len(training) else set()
        pf["novel_scaffold"] = (~pf["chemotype_scaffold"].isin(train_scaffolds)
                                & (pf["chemotype_scaffold"] != "")).astype(int)

        # families = distinct Bemis-Murcko scaffolds
        fam = pf.groupby("chemotype_scaffold", as_index=False).agg(
            n_candidates=("molecule_id", "count"),
            n_portfolio_groups=("portfolio_group", "nunique"),
            novel_scaffold=("novel_scaffold", "first"),
            representative_molecule_id=("molecule_id", "first"),
            mean_predicted_activity=("predicted_activity", "mean"),
            mean_novelty=("novelty", "mean"),
            mean_uncertainty=("uncertainty", "mean"),
            mean_information_gain=("information_gain_score", "mean"),
            mean_taf_score=("TAF_score_2d", "mean"),
            aromatic_rings=("mol", lambda s: int(np.mean([_arom_rings(m) for m in s]))),
        )
        fam = fam[fam["chemotype_scaffold"] != ""].reset_index(drop=True)

        # per-molecule library annotations (drug-likeness + PAINS for the plates)
        dl = pf["mol"].apply(_druglike_flags)
        pf = pd.concat([pf, pd.DataFrame(dl.tolist())], axis=1)
        pf["druglike"] = ((pf["pains_clean"] == 1)
                          & (pf["brenk_clean"] == 1)
                          & (pf["lipinski_violations"] <= 2)).astype(int)

        # tier rule (documented, deterministic) - tier1 requires clean novelty
        tier1_groups = {"A_high_confidence_taf_preserving", "B_taf_disrupting",
                        "C_scaffold_transfer", "E_information_gain"}
        pf["molecule_tier"] = np.where(
            (pf["novel_scaffold"] == 1) & (pf["druglike"] == 1), 1,
            np.where(pf["portfolio_group"].isin(tier1_groups), 2, 3))
        fam_tier = pf.assign(
            scaffold=pf["chemotype_scaffold"]
        ).groupby("scaffold")["molecule_tier"].min().reset_index()
        fam_tier.columns = ["chemotype_scaffold", "tier"]
        fam = fam.merge(fam_tier, on="chemotype_scaffold", how="left")
        fam["tier"] = fam["tier"].fillna(3).astype(int)
        fam = fam.sort_values(["tier", "n_candidates"],
                              ascending=[True, False]).reset_index(drop=True)
        fam["family_id"] = [f"CHT-{i+1:03d}" for i in range(len(fam))]
        fam["priority_rank"] = range(1, len(fam) + 1)
        pf["family_id"] = pf["chemotype_scaffold"].map(
            {r["chemotype_scaffold"]: r["family_id"] for _, r in fam.iterrows()})

        # round-wise coverage targets (pre-registered, go/no-go style)
        n_rounds = int(self.cfg.raw.get("prospective", {}).get("rounds", 3))
        per_round = int(self.cfg.raw.get("prospective", {}).get("per_round_n", 15))
        cap_per_plate = max(2, int(np.ceil(len(fam) / max(1, n_rounds))))
        n_families = len(fam)
        n_novel = int(fam["novel_scaffold"].sum())
        targets = pd.DataFrame({
            "round": list(range(1, n_rounds + 1)),
            "target_new_chemotypes_per_round": [min(cap_per_plate, per_round)] * n_rounds,
            "scaffold_cap_per_plate": [2] * n_rounds,
            "target_clean_novel_chemotypes_added": [
                min(cap_per_plate, max(1, int(fam[(fam["tier"] == 1)].shape[0]) // n_rounds))
            ] * n_rounds,
            "primary_hit_criteria": ["potency>0 AND potentiation>0 confirmed by dose-response"] * n_rounds,
            "enrichment_factor_threshold": [2.0] * n_rounds,
            "go_no_go": ["continue if EF>=2.0 at top-k and >=1 clean novel chemotype confirmed"] * n_rounds,
        })

        # external validation status (honest)
        ext_dir = self.cfg.resolve("data/external")
        ext_files = [str(p) for p in ext_dir.rglob("*") if p.is_file()] if ext_dir.exists() else []
        ext_status = {
            "external_datasets_found": len(ext_files),
            "source": "search of data/external for published a9a10 / nicotinic PAM datasets",
            "validation": ("models NOT yet validated on external data" if not ext_files
                           else "external validation pending downstream run"),
            "note": "no fabricated external data; enrichment claims rest on internal data until provided",
        }

        save_df(fam, str(self.cfg.resolve("loop/library/chemotype_families.csv")))
        save_df(pf, str(self.cfg.resolve("loop/library/chemotype_library_annotated.csv")))
        save_df(targets, str(self.cfg.resolve("loop/library/coverage_targets.csv")))
        save_manifest(ext_status, str(self.cfg.resolve("loop/library/external_validation_status.json")))
        save_manifest(
            make_provenance("library", self.cfg, {
                "n_families": n_families, "n_novel_scaffolds": n_novel,
                "tier_counts": obj2int(fam["tier"].value_counts().to_dict()),
                "tier_rule": TIER_RULE, "external": ext_status,
            }),
            str(self.cfg.resolve("loop/library/library_manifest.json")),
        )
        self.log.info(f"Chemotype families: {n_families} "
                      f"(novel scaffolds: {n_novel})")
        self.log.info(fam[["family_id", "chemotype_scaffold", "tier",
                           "n_candidates", "novel_scaffold"]].head(10).to_string(index=False))
        return {"families": fam, "annotated": pf, "targets": targets}


def obj2int(d: Dict[str, Any]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in d.items()}


def run_phase_library(cfg_path: str,
                      portfolio: pd.DataFrame = None,
                      training: pd.DataFrame = None) -> Dict[str, Any]:
    import os
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("library", str(cfg.resolve("logs")))
    if portfolio is None:
        p = str(cfg.resolve("prospective/prospective_candidates.csv"))
        portfolio = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    if training is None:
        p = str(cfg.resolve("data/processed/data_analysis_ready.csv"))
        training = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    lib = ChemotypeLibrary(cfg, logger)
    return lib.project(portfolio, training)


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase_library(cfgp)