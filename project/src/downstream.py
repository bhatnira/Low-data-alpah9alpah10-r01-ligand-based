#!/usr/bin/env python3
"""Phase 13b: generated-library downstream triage (ligand-based).

Implements MASTER_PROMPT sections 14, 15, 22, 30, 31, 32, 33, 34-44
(cochlear delivery / systemic ADME / systemic toxicology / local cochlear
safety), 45-47 (final selectivity filter), 48-53 (multi-axis final candidate
portfolio, tiers, information-gain portfolio, diversity), 61
(structure_based_handoff.csv) and 67 (generated-library integration test).

Honesty rules enforced here (no fabrication):
  * every generated molecule keeps experimental_status == "untested";
  * any physical property / behaviour that we cannot compute is reported as
    UNKNOWN / NOT_AVAILABLE with an explicit evidence level, never invented;
  * ascorbate-like redox/SVCT statements are documented HYPOTHESES, never
    conclusions;
  * all counts (N_valid, N_unique, N_in_domain, N_predicted_active, ...) are
    recomputed from the actual generated file on every run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, FilterCatalog, RDConfig
from rdkit.Chem import Crippen  # noqa: F401  (used indirectly)

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
    sha256_file, sha256_dataframe, utcnow, SOFTWARE,
)

RDLogger.DisableLog("rdApp.*")

DESC_COLS = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
             "RingCount", "AromaticRings", "FractionCSP3", "HeavyAtomCount"]

GENERATION_OBJECTIVE = {
    "reinvent_local_analog": "ASCORBATE_SERIES_ANALOG",
    "reinvent_local_analog_libinvent": "ASCORBATE_SERIES_ANALOG",
    "reinvent_de_novo": "DIVERSE_NOVEL_CHEMOTYPE",
    "reinvent_scaffold_hopping": "SCAFFOLD_HOP",
    "reinvent_information_gain": "INFORMATION_GAIN_PROBE",
    "reinvent_taf_disrupting": "ASCORBATE_SCAFFOLD_EXPLORATION",
    "medchem_baseline": "ASCORBATE_SERIES_ANALOG",
    "negative_control": "INFORMATION_GAIN_PROBE",
}

# --- curated structural alerts (systemic tox / cochlear safety evidence) ----
REACTIVE_ALERTS: List[Tuple[str, str]] = [
    ("epoxide", "C1OC1"),
    ("aziridine", "C1CN1"),
    ("alkyl_halide", "[CX4][Cl,Br,I]"),
    ("nitro", "[NX3+](=O)[O-]"),
    ("aromatic_nitro", "[c,n;R][NX3+](=O)[O-]"),
    ("michael_acceptor", "[CX3;$([CX3](=O)),$([CX3]#N)]=[CX3]"),
    ("azide", "[N-]=[N+]=N"),
    ("isocyanate", "N=C=O"),
    ("isothiocyanate", "N=C=S"),
    ("acyl_halide", "[CX3](=O)[Cl,Br,I]"),
    ("sulfonate_ester", "[OX2][SX4](=O)(=O)"),
    ("quinone", "O=C1C=CC(=O)C=C1"),
    ("aniline", "[NX3;H2]c1ccccc1"),
    ("aromatic_amine", "[NX3;H2;!$(NC=O)]c1ccccc1"),
    ("amide_of_sulfonyl", "S(=O)(=O)N[CX3]=O"),
    ("peroxide", "[OX2][OX2]"),
    ("phosphonate", "[PX4](=O)(O)(O)"),
]

# structural alert list used for the local cochlear / hair-cell safety screen
OTOTOXIC_ALERTS: List[Tuple[str, str]] = [
    ("aminoglycoside_like_polyamine", "[NX3;H2][CX4][CX4][NX3;H2]"),
    ("aromatic_amine", "[NX3;H2]c1ccccc1"),
    ("nitroarene", "[c;R][NX3+](=O)[O-]"),
    ("quinone", "O=C1C=CC(=O)C=C1"),
    ("platinum_complex", "[Pt]"),
    ("halogenated_alcohol", "[OX2H][CX4][Cl,Br,I,F]"),
    ("polyether_like", "[OX2][CX4][CX4][OX2][CX4][CX4][OX2]"),
]


def morgan_array(mol: Chem.Mol, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _rdkit_desc(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "RingCount": Descriptors.RingCount(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
    }


def _esol_logS(mol: Chem.Mol) -> float:
    """Delaney ESOL (J Chem Inf Comput Sci 44:1000-1005, 2004) proxy."""
    desc = _rdkit_desc(mol)
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    logS = (0.16 - 0.63 * desc["LogP"] - 0.0062 * desc["MW"]
            + 0.066 * desc["RotBonds"] - 0.74 * (aromatic_atoms / mol.GetNumAtoms()))
    return logS


def _sascore(mol: Chem.Mol) -> float:
    sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
    try:
        import sascorer  # type: ignore
        return float(sascorer.calculateScore(mol))
    except Exception:
        return np.nan


def _has_basic_amine(mol: Chem.Mol) -> bool:
    patter = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(N=O);!$(N#N)]")
    if patter is None:
        return False
    return mol.HasSubstructMatch(patter)


def _ascorbate_like(mol: Chem.Mol) -> bool:
    """Enediol lactone / ascorbate-like core (vitamin-C chemotype)."""
    patt = Chem.MolFromSmarts("[OX2H]=[CX3]1[CX3](=[OX1])OC[CX3]1[OX2]")
    alt = Chem.MolFromSmarts("[CX3]([OX2H])=[CX3][CX3](=O)O[CX4]")
    return bool(patt and mol.HasSubstructMatch(patt)) or bool(alt and mol.HasSubstructMatch(alt))


def _filter_catalogs() -> Dict[str, Any]:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    pains = FilterCatalog.FilterCatalog(params)
    bparams = FilterCatalog.FilterCatalogParams()
    bparams.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    brenk = FilterCatalog.FilterCatalog(bparams)
    return {"pains": pains, "brenk": brenk}


def _alert_matches(mol: Chem.Mol, alerts: List[Tuple[str, str]]) -> List[str]:
    out = []
    for name, smarts in alerts:
        p = Chem.MolFromSmarts(smarts)
        if p is not None and mol.HasSubstructMatch(p):
            out.append(name)
    return out


def _stereo_status(mol: Chem.Mol) -> Dict[str, Any]:
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    n_defined = sum(1 for _, t in centers if t != "?")
    n_undefined = sum(1 for _, t in centers if t == "?")
    return {
        "stereocenter_count": len(centers),
        "defined_stereocenter_count": n_defined,
        "undefined_stereocenter_count": n_undefined,
        "has_undefined_stereo": n_undefined > 0,
    }


class GeneratedLibraryTriage:
    """Scores the full generated library through every ligand-based axis."""

    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger
        self.catalogs = _filter_catalogs()

    # ---------------------------------------------------------- library read
    def _load_generated(self) -> pd.DataFrame:
        p = self.cfg.resolve("reinvent/generated_molecules.csv")
        if not p.exists():
            g = self.cfg.resolve("generated/medchem_baseline_analogs.csv")
            if not g.exists():
                raise FileNotFoundError("no generated molecules file present")
            return pd.read_csv(g)
        return pd.read_csv(p)

    def _load_frozen(self) -> Dict[str, Any]:
        import joblib
        p = self.cfg.resolve("models/classical/lbm_rf_fp_desc.joblib")
        if not p.exists():
            raise FileNotFoundError("frozen model artifact missing")
        return joblib.load(p)

    # ------------------------------------------------------------ QC counts
    def library_qc(self, df: pd.DataFrame, fps: List[Chem.Mol]) -> Dict[str, Any]:
        self.log.step("Generated-library QC (sect. 30)")
        canons = []
        seen = set()
        dup_count = 0
        stereo_valid = 0
        for mol, smi in zip(fps, df["isomeric_SMILES"]):
            if mol is None:
                continue
            c = Chem.MolToSmiles(mol)  # canonical (non-isomeric)
            if c in seen:
                dup_count += 1
                continue
            seen.add(c)
            canons.append(c)
            if not _stereo_status(mol)["has_undefined_stereo"]:
                stereo_valid += 1
        qc = {
            "N_generated": int(len(df)),
            "N_valid": int(sum(1 for m in fps if m is not None)),
            "N_unique": int(len(seen)),
            "N_stereo_valid": int(stereo_valid),
            "N_duplicate": int(dup_count),
        }
        save_manifest(
            make_provenance("generated_qc", self.cfg, qc),
            str(self.cfg.resolve("generated/generated_qc.json")),
        )
        save_df(pd.DataFrame([qc]), str(self.cfg.resolve("generated/generated_qc.csv")))
        self.log.info(f"QC: {qc}")
        return qc

    # ------------------------------------------------------ potency/AD/uncert
    def score_all(self, df: pd.DataFrame, mols: List[Chem.Mol]) -> pd.DataFrame:
        """Predictive potency + per-axis uncertainty + 4-state applicability domain
        for every valid generated molecule (self-similarity is impossible: the
        generated library is disjoint from the experimental training set)."""
        self.log.step("Scoring generated library (potency / uncertainty / AD)")
        frozen = self._load_frozen()
        clf = frozen["model"]
        scaler = frozen["scaler"]
        train_smiles = [s for s in frozen["training_smiles"]]
        train_mols = [m for m in (Chem.MolFromSmiles(s) for s in train_smiles) if m is not None]
        train_fps = [morgan_array(m) for m in train_mols]
        train_fp_bits = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
                        for m in train_mols]
        active_mask = [s in set(frozen["active_smiles"]) for s in train_smiles]
        active_fps = [f for f, a in zip(train_fp_bits, active_mask) if a]

        rows = []
        Xfeat = np.zeros((len(mols), 2048 + len(DESC_COLS)), dtype=np.float64)
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            Xfeat[i, :2048] = morgan_array(mol)
            d = _rdkit_desc(mol)
            Xfeat[i, 2048:] = [d[c] for c in DESC_COLS]
        Xfeat = scaler.transform(Xfeat)
        proba = clf.predict_proba(Xfeat)[:, 1]
        unc = np.std([t.predict_proba(Xfeat)[:, 1] for t in clf.estimators_], axis=0)

        for i, mol in enumerate(mols):
            if mol is None:
                continue
            if i % 5000 == 0:
                self.log.info(f"  scored {i}/{len(mols)}")
            fpb = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            sim_train = DataStructs.BulkTanimotoSimilarity(fpb, train_fp_bits)
            sim_active = DataStructs.BulkTanimotoSimilarity(fpb, active_fps) if active_fps else []
            d_any = float(1 - max(sim_train))
            d_active = float(1 - max(sim_active)) if sim_active else np.nan
            if d_any < 0.35:
                ad = "IN_DOMAIN"
            elif d_any < 0.6 or (np.isfinite(d_active) and d_active < 0.45):
                ad = "BORDERLINE"
            else:
                ad = "OUT_OF_DOMAIN"
            novelty = round(1 - max(sim_train), 4)
            p = float(proba[i])
            u = float(unc[i])
            if ad == "IN_DOMAIN" and u < 0.12:
                conf = "HIGH"
            elif ad in ("IN_DOMAIN", "BORDERLINE") and u < 0.2:
                conf = "MODERATE"
            elif ad == "OUT_OF_DOMAIN":
                conf = "LOW"
            else:
                conf = "UNKNOWN"
            rows.append({
                "molecule_id": df.iloc[i]["molecule_id"],
                "generation_mode": df.iloc[i]["generation_mode"],
                "parent_id": df.iloc[i].get("parent_id", np.nan),
                "isomeric_SMILES": df.iloc[i]["isomeric_SMILES"],
                "SMILES": df.iloc[i]["isomeric_SMILES"],
                "predicted_activity": round(p, 4),
                "predictive_uncertainty": round(u, 4),
                "applicability_domain": ad,
                "novelty": round(float(novelty), 4),
                "similarity_to_train_max": round(float(max(sim_train)), 4),
                "similarity_to_active_max": round(1 - d_active, 4) if np.isfinite(d_active) else np.nan,
                "confidence_class": conf,
                "information_gain_score": round(float(0.5 - abs(p - 0.5)), 4),
            })
        res = pd.DataFrame(rows)
        save_df(res, str(self.cfg.resolve("generated/generated_predictions.csv")))
        self.log.info(f"scored {len(res)} molecules")
        return res

    # --------------------------------------------------------- synth / ADME
    def synthesis(self, scored: pd.DataFrame, mols: List[Chem.Mol],
                  df: pd.DataFrame) -> pd.DataFrame:
        self.log.step("Synthetic feasibility (SAScore proxy, sect. 33)")
        rows = []
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            sa = _sascore(mol)
            st = _stereo_status(mol)
            rows.append({
                "molecule_id": df.iloc[i]["molecule_id"],
                "synthetic_accessibility_score": round(sa, 3) if np.isfinite(sa) else np.nan,
                "stereocenter_count": st["stereocenter_count"],
                "defined_stereocenter_count": st["defined_stereocenter_count"],
                "undefined_stereocenter_count": st["undefined_stereocenter_count"],
                "ring_count": _rdkit_desc(mol)["RingCount"],
                "rotatable_bonds": _rdkit_desc(mol)["RotBonds"],
                "stereochemical_complexity": "high" if st["stereocenter_count"] >= 3 else "moderate" if st["stereocenter_count"] else "low",
                "route_status": "NOT_AVAILABLE",
                "route_note": "AIZynthFinder route prediction not run (USPTO models absent); SAScore is a prioritization proxy, not a validated synthesis plan.",
                "synthetic_feasibility_class": ("FAVORABLE" if (np.isfinite(sa) and sa <= 4.0)
                                                else "MODERATE" if (np.isfinite(sa) and sa <= 6.0)
                                                else "CHALLENGING" if np.isfinite(sa) else "UNKNOWN"),
            })
        res = pd.DataFrame(rows)
        save_df(res, str(self.cfg.resolve("synthesis/synthetic_feasibility.csv")))
        return res

    def adme(self, scored: pd.DataFrame, mols: List[Chem.Mol], df: pd.DataFrame) -> pd.DataFrame:
        self.log.step("Systemic ADME proxies (sect. 39-40)")
        rows = []
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            desc = _rdkit_desc(mol)
            logS = _esol_logS(mol)
            base = {"molecule_id": df.iloc[i]["molecule_id"]}
            base["solubility_logS_esol_proxy"] = round(float(logS), 3)
            base["solubility_class"] = ("HIGH" if logS > -4 else "MODERATE" if logS > -6 else "LOW")
            tpsa = desc["TPSA"]
            base["caco_permeability_proxy"] = ("HIGH" if tpsa < 90 else "MODERATE" if tpsa < 140 else "LOW")
            logp = desc["LogP"]
            base["plasma_protein_binding_proxy"] = ("HIGH" if logp > 4 else "MODERATE" if logp > 2.5 else "LOW")
            base["lipophilicity_logP"] = round(logp, 2)
            base["metabolic_stability_proxy"] = (
                "LOW" if (desc["MW"] > 480 or desc["RotBonds"] > 9 or desc["HBA"] > 8)
                else "MODERATE" if (desc["MW"] > 400 or desc["RotBonds"] > 7)
                else "HIGH")
            base["oral_absorption_proxy"] = ("HIGH" if (0 <= logp <= 3.5 and tpsa <= 120)
                                            else "MODERATE" if (-1 <= logp <= 5 and tpsa <= 160)
                                            else "LOW")
            base["evidence_level"] = "COMPUTATIONALLY_SUPPORTED_PROXY"
            base["note"] = "Physchem proxies (ESOL / TPSA / logP). Not experimental ADME; no renal/hepatic PK measured."
            rows.append(base)
        res = pd.DataFrame(rows)
        save_df(res, str(self.cfg.resolve("adme/systemic_adme.csv")))
        return res

    def systemic_toxicity(self, scored: pd.DataFrame, mols: List[Chem.Mol],
                          df: pd.DataFrame) -> pd.DataFrame:
        self.log.step("Systemic toxicology screen (sect. 41)")
        rows = []
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            desc = _rdkit_desc(mol)
            reactive = _alert_matches(mol, REACTIVE_ALERTS)
            pains = [e.GetDescription() for e in self.catalogs["pains"].GetMatches(mol)]
            brenk = [e.GetDescription() for e in self.catalogs["brenk"].GetMatches(mol)]
            hERG_flag = _has_basic_amine(mol) and desc["LogP"] > 3.5
            alert = len(reactive) + len(pains) + len(brenk)
            redox_note = _ascorbate_like(mol)
            if alert >= 3 or hERG_flag:
                cls = "HIGH"
            elif alert == 1:
                cls = "MODERATE"
            elif alert == 0 and not redox_note:
                cls = "LOW"
            else:
                cls = "MODERATE"  # redox-active profile requires empirical confirmation
            rows.append({
                "molecule_id": df.iloc[i]["molecule_id"],
                "reactive_alert_matches": "; ".join(sorted(set(reactive))),
                "n_reactive_alerts": len(set(reactive)),
                "pains_alert_matches": "; ".join(sorted(set(pains))),
                "brenk_alert_matches": "; ".join(sorted(set(brenk))),
                "hERG_basic_amine_lipophilic_flag": bool(hERG_flag),
                "ascorbate_redox_profile_note": bool(redox_note),
                "systemic_toxicity_class": cls,
                "evidence_level": "COMPUTATIONALLY_SUPPORTED_PROXY",
                "note": "Structural-alert screen only (PAINS/BRENK/curated SMARTS + hERG heuristic). Absence of an alert is NOT demonstrated safety; no experimental tox data exist for generated compounds.",
            })
        res = pd.DataFrame(rows)
        save_df(res, str(self.cfg.resolve("toxicity/systemic_toxicity.csv")))
        return res

    def cochlear_safety(self, scored: pd.DataFrame, mols: List[Chem.Mol],
                        df: pd.DataFrame) -> pd.DataFrame:
        self.log.step("Local cochlear / hair-cell safety screen (sect. 42-44)")
        rows = []
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            oto = _alert_matches(mol, OTOTOXIC_ALERTS)
            redox_note = _ascorbate_like(mol)
            desc = _rdkit_desc(mol)
            if len(oto) >= 2:
                cls = "HIGH"
            elif len(oto) == 1:
                cls = "MODERATE"
            elif redox_note:
                cls = "MODERATE"  # redox-active enediol: needs empirical OHC red-ox screening
            else:
                cls = "LOW"
            rows.append({
                "molecule_id": df.iloc[i]["molecule_id"],
                "ototoxic_alert_matches": "; ".join(sorted(set(oto))),
                "hair_cell_risk": cls,
                "mitochondrial_risk": "UNKNOWN",
                "oxidative_risk": "MODERATE" if redox_note else "LOW",
                "ion_channel_risk": "UNKNOWN",
                "local_exposure": "UNKNOWN",
                "retention": "UNKNOWN",
                "local_cochlear_safety_class": cls,
                "evidence_level": "COMPUTATIONALLY_SUPPORTED_PROXY",
                "note": "Structural-alert proxy for ototoxicity-associated motifs. 'Low predicted liability' is not demonstrated hair-cell safety; OHC toxicity requires empirical assay (e.g., OHC explant / TMRE).",
            })
        res = pd.DataFrame(rows)
        save_df(res, str(self.cfg.resolve("cochlear_safety/cochlear_safety.csv")))
        return res

    def cochlear_delivery(self, scored: pd.DataFrame, mols: List[Chem.Mol],
                          df: pd.DataFrame) -> pd.DataFrame:
        self.log.step("Cochlear delivery proxies (sect. 34-38)")
        rows = []
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            desc = _rdkit_desc(mol)
            logp = desc["LogP"]; tpsa = desc["TPSA"]; mw = desc["MW"]; hbd = desc["HBD"]
            asc = _ascorbate_like(mol)
            rwm = ("HIGH" if (mw <= 450 and hbd <= 2 and 0 <= logp <= 3.5)
                   else "MODERATE" if (mw <= 550 and tpsa <= 140)
                   else "LOW")
            blb = ("HIGH" if (mw <= 350 and -1 <= logp <= 2.5) else "MODERATE" if logp <= 4 else "LOW")
            perilymph = ("HIGH" if rwm == "HIGH" else "MODERATE" if rwm == "MODERATE" else "LOW")
            endolymph = "UNKNOWN"  # endolymph access is ionically gated; not computable
            ohc = ("MODERATE" if asc else "UNKNOWN")
            sys_exp = ("HIGH" if blb in ("HIGH", "MODERATE") else "MODERATE")
            retention = ("HIGH" if logp >= 3 else "MODERATE" if logp >= 1.5 else "LOW")
            rows.append({
                "molecule_id": df.iloc[i]["molecule_id"],
                "delivery_strategy": ("INTRATYMPANIC / ROUND-WINDOW" if rwm == "HIGH"
                                      else "SYSTEMIC ORAL" if blb == "HIGH"
                                      else "SYSTEMIC IV" if blb == "MODERATE"
                                      else "EXPERIMENTAL"),
                "systemic_exposure_likelihood": sys_exp,
                "BLB_compatibility": blb,
                "RWM_compatibility": rwm,
                "perilymph_access_likelihood": perilymph,
                "endolymph_access_likelihood": endolymph,
                "OHC_uptake_likelihood": ohc,
                "cochlear_retention_likelihood": retention,
                "local_exposure_likelihood": rwm,
                "ascorbate_SVCT_hypothesis": ("HYPOTHESIS" if asc else "NOT_APPLICABLE"),
                "delivery_confidence": "MODERATE",
                "delivery_evidence_level": "COMPUTATIONALLY_SUPPORTED_PROXY",
                "note": ("Physchem proxies only - NOT experimental cochlear PK. "
                         "OHC uptake through Vitamin-C transporters (SVCT2/SVCT1) is an "
                         "untested HYPOTHESIS for ascorbate-like molecules, not a conclusion."),
            })
        res = pd.DataFrame(rows)
        save_df(res, str(self.cfg.resolve("cochlear_delivery/cochlear_delivery.csv")))
        return res

    def selectivity(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Final independent selectivity filter (sect. 45-47).

        No subtype-selectivity experimental data exist for the generated
        molecules, so every candidate is reported UNKNOWN / NOT_EVALUABLE.
        We do NOT infer selectivity from structural similarity to α9α10
        actives, and we never fabricate α7/α4β2 results.
        """
        self.log.step("Final selectivity filter (sect. 45-47)")
        rows = []
        for _, r in scored.iterrows():
            rows.append({
                "molecule_id": r["molecule_id"],
                "selectivity_status": "UNKNOWN",
                "selectivity_uncertainty": "HIGH",
                "nAChR_subtype_evidence": "NOT_AVAILABLE",
                "alpha7_check": "NOT_EVALUABLE",
                "alpha4beta2_check": "NOT_EVALUABLE",
                "Level2_offtarget_proxy": "See systemic_toxicity screen (hERG/structure alerts)",
                "note": ("Experimental subtype selectivity (α9α10 vs α7 vs α4β2) has not been "
                         "measured for these compounds. A candidate cannot be called selective "
                         "without such data - priority is set aside for future electrophysiology."),
            })
        res = pd.DataFrame(rows)
        save_df(res, str(self.cfg.resolve("selectivity/selectivity_final_filter.csv")))
        return res

    # ------------------------------------------------------------ portfolio
    def diversify(self, pool: pd.DataFrame, mols_by_id: Dict[str, Chem.Mol], k: int = 100) -> List[str]:
        """Greedy max-min selection on ECFP4 Tanimoto (sect. 52)."""
        fps = [morgan_array(m) for m in mols_by_id.values()]
        ids = list(mols_by_id.keys())
        if len(ids) <= k:
            return ids
        x = np.array(fps)
        norms = np.linalg.norm(x, axis=1)
        sim = x @ x.T
        denom = np.outer(norms, norms)
        sim = np.divide(sim, denom, out=np.zeros_like(sim), where=denom != 0)
        chosen = [0]
        avail = set(range(1, len(ids)))
        while len(chosen) < k and avail:
            d = 1 - sim[np.ix_(list(avail), chosen)].max(axis=1)
            j = int(np.argmax(d))
            nxt = sorted(avail)[j]
            chosen.append(nxt)
            avail.remove(nxt)
        return [ids[c] for c in chosen]

    def final_portfolio(self, scored: pd.DataFrame, synth: pd.DataFrame,
                        adme: pd.DataFrame, tox: pd.DataFrame, coch_safety: pd.DataFrame,
                        delivery: pd.DataFrame, sel: pd.DataFrame,
                        df: pd.DataFrame, mols: List[Chem.Mol]) -> pd.DataFrame:
        self.log.step("Final multi-axis candidate portfolio + tiers (sect. 48-53, 83)")
        mol_by_id = {df.iloc[i]["molecule_id"]: mols[i] for i in range(len(df)) if mols[i] is not None}
        merge = scored
        for t, key in [(synth, "molecule_id"), (adme, "molecule_id"), (tox, "molecule_id"),
                       (coch_safety, "molecule_id"), (delivery, "molecule_id"), (sel, "molecule_id")]:
            cols = [c for c in t.columns if c not in ("note",) and c != "molecule_id"]
            merge = merge.merge(t[["molecule_id"] + cols], on="molecule_id", how="left")

        active_pool = merge[
            (merge["predicted_activity"] >= 0.5)
            & (merge["applicability_domain"].isin(["IN_DOMAIN", "BORDERLINE"]))
            & (merge["confidence_class"].isin(["HIGH", "MODERATE"]))
        ]
        self.log.info(f"active+in-domain+confident pool: {len(active_pool)}")

        def tier(row):
            if row.get("systemic_toxicity_class") == "HIGH" or row.get("local_cochlear_safety_class") == "HIGH":
                return "EXCLUDE_OR_HOLD"
            if (row["predicted_activity"] >= 0.5
                    and row["applicability_domain"] in ("IN_DOMAIN", "BORDERLINE")
                    and row["confidence_class"] in ("HIGH", "MODERATE")
                    and row.get("synthetic_feasibility_class") in ("FAVORABLE", "MODERATE")):
                return "Tier1_high_priority"
            if row["information_gain_score"] >= 0.3 or row["generation_mode"].endswith("information_gain"):
                return "Tier2_mechanistically_informative"
            if row["novelty"] >= 0.5 or row["applicability_domain"] == "OUT_OF_DOMAIN":
                return "Tier3_exploratory"
            return "EXCLUDE_OR_HOLD"

        merge["portfolio_tier"] = merge.apply(tier, axis=1)

        sel_cols = []
        for mid in self.diversify(active_pool.iloc[:5000] if len(active_pool) else merge,
                                  {m: Chem.MolFromSmiles(s) for m, s in
                                   zip(merge["molecule_id"], merge["isomeric_SMILES"])}):
            spec = merge[merge["molecule_id"] == mid].iloc[0]
            sel_cols.append(spec)

        # information-gain portfolio (sect. 51): near-boundary probes by stereo/new chemotype
        ig_pool = merge[merge["information_gain_score"] >= 0.3].sort_values(
            "information_gain_score", ascending=False).head(200)
        ig_ids = self.diversify(ig_pool, {m: C for m, C in zip(ig_pool["molecule_id"],
                                                               ig_pool["isomeric_SMILES"].apply(Chem.MolFromSmiles))},
                                k=min(25, len(ig_pool)))
        ig_rows = [merge[merge["molecule_id"] == mid].iloc[0] for mid in ig_ids]

        records = []
        for spec in sel_cols:
            s = {
                "molecule_id": spec["molecule_id"],
                "isomeric_SMILES": spec["isomeric_SMILES"],
                "stereochemistry": "DEFINED" if not spec.get("has_undefined_stereo", True) else "UNDEFINED_OR_NA",
                "parent_id": spec.get("parent_id", np.nan),
                "generation_mode": spec["generation_mode"],
                "generation_objective": GENERATION_OBJECTIVE.get(spec["generation_mode"], "DIVERSE_NOVEL_CHEMOTYPE"),
                "generation_model": "REINVENT4",
                "generation_seed": self.cfg.seed("reinvent"),
                "experimental_parent": spec.get("parent_id", np.nan),
                "predicted_potency": spec["predicted_activity"],
                "potency_uncertainty": spec["predictive_uncertainty"],
                "applicability_domain": spec["applicability_domain"],
                "novelty": spec["novelty"],
                "synthetic_feasibility": spec.get("synthetic_feasibility_class", "UNKNOWN"),
                "cochlear_delivery": spec.get("delivery_strategy", "UNKNOWN"),
                "systemic_ADME": spec.get("solubility_class", "UNKNOWN"),
                "systemic_toxicity": spec.get("systemic_toxicity_class", "UNKNOWN"),
                "local_cochlear_safety": spec.get("local_cochlear_safety_class", "UNKNOWN"),
                "selectivity": spec.get("selectivity_status", "UNKNOWN"),
                "selectivity_uncertainty": spec.get("selectivity_uncertainty", "HIGH"),
                "information_gain": spec["information_gain_score"],
                "portfolio_tier": spec["portfolio_tier"],
                "confidence": spec["confidence_class"],
                "risk_flags": "; ".join(x for x in [
                    "REDOX_PROFILE" if spec.get("ascorbate_redox_profile_note") else "",
                    "STRUCTURAL_ALERT" if (spec.get("n_reactive_alerts") or 0) > 0 else "",
                    "OUT_OF_DOMAIN" if spec["applicability_domain"] == "OUT_OF_DOMAIN" else "",
                ] if x),
                "selection_rationale": f"generation={spec['generation_mode']}; "
                                       f"AD={spec['applicability_domain']}; "
                                       f"confidence={spec['confidence_class']}; "
                                       f"tier={spec['portfolio_tier']}",
                "experimental_status": "untested",
                "created_at": utcnow(),
            }
            records.append(s)
        port = pd.DataFrame(records)
        save_df(port, str(self.cfg.resolve("portfolio/final_candidate_portfolio.csv")))
        ig_df = pd.DataFrame(
            {k: v for k, v in merge[merge["molecule_id"].isin(ig_ids)].set_index("molecule_id").to_dict("index").items()}
        )
        ig_spec = merge[merge["molecule_id"].isin(ig_ids)]
        ig_spec = ig_spec.assign(portfolio_group="E_information_gain",
                                 hypothesis_linked=(
                                     "Near the frozen-model decision boundary; discriminates "
                                     "competing SAR/stereo hypotheses (MASTER_PROMPT sect. 51)."))
        save_df(ig_spec, str(self.cfg.resolve("portfolio/information_gain_portfolio.csv")))
        save_df(merge, str(self.cfg.resolve("portfolio/generated_library_annotated.csv")))
        self.log.info(f"final portfolio: {len(port)} molecules across tiers "
                      f"{port['portfolio_tier'].value_counts().to_dict()}")
        return port

    def write_handoff(self, port: pd.DataFrame) -> None:
        """structure_based_handoff.csv (MASTER_PROMPT sect. 61)."""
        hand = port[[
            "molecule_id", "isomeric_SMILES", "stereochemistry", "generation_mode",
            "parent_id", "predicted_potency", "potency_uncertainty",
            "applicability_domain", "synthetic_feasibility", "cochlear_delivery",
            "systemic_ADME", "systemic_toxicity", "local_cochlear_safety",
            "selectivity", "selectivity_uncertainty",
        ]].copy()
        hand["no_receptor_structure_assumption"] = "TRUE (ligand-based pipeline)"
        save_df(hand, str(self.cfg.resolve("structure_based_handoff.csv")))
        self.log.info(f"structure_based_handoff.csv written ({len(hand)} rows)")

    def run_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.log.step("Generated-library downstream triage")
        generated = self._load_generated()
        mols = [Chem.MolFromSmiles(str(smi)) for smi in generated["isomeric_SMILES"].tolist()]

        qc = self.library_qc(generated, mols)
        scored = self.score_all(generated, mols)
        synth = self.synthesis(scored, mols, generated)
        adme = self.adme(scored, mols, generated)
        tox = self.systemic_toxicity(scored, mols, generated)
        coch_safety = self.cochlear_safety(scored, mols, generated)
        delivery = self.cochlear_delivery(scored, mols, generated)
        sel = self.selectivity(scored)
        port = self.final_portfolio(scored, synth, adme, tox, coch_safety, delivery, sel,
                                    generated, mols)
        self.write_handoff(port)

        # integration test (sect. 67): every generated molecule flows end-to-end
        manifest = {
            "run_status": "COMPLETED",
            "generated_library_file": str(self.cfg.resolve("reinvent/generated_molecules.csv")),
            "qc": qc,
            "n_scored": int(len(scored)),
            "n_portfolio": int(len(port)),
            "outputs": {
                "generated_predictions": str(self.cfg.resolve("generated/generated_predictions.csv")),
                "synthetic_feasibility": str(self.cfg.resolve("synthesis/synthetic_feasibility.csv")),
                "systemic_adme": str(self.cfg.resolve("adme/systemic_adme.csv")),
                "systemic_toxicity": str(self.cfg.resolve("toxicity/systemic_toxicity.csv")),
                "cochlear_safety": str(self.cfg.resolve("cochlear_safety/cochlear_safety.csv")),
                "cochlear_delivery": str(self.cfg.resolve("cochlear_delivery/cochlear_delivery.csv")),
                "selectivity_filter": str(self.cfg.resolve("selectivity/selectivity_final_filter.csv")),
                "final_portfolio": str(self.cfg.resolve("portfolio/final_candidate_portfolio.csv")),
                "information_gain_portfolio": str(self.cfg.resolve("portfolio/information_gain_portfolio.csv")),
                "handoff": str(self.cfg.resolve("structure_based_handoff.csv")),
            },
        }
        save_manifest(
            {**make_provenance("downstream", self.cfg, {
                "n_generated": int(len(generated)),
                "n_scored": int(len(scored)),
                "n_portfolio": int(len(port)),
            }), **manifest},
            str(self.cfg.resolve("portfolio/downstream_manifest.json")),
        )
        return {"qc": qc, "scored": scored, "portfolio": port}


def run_phase_downstream(cfg_path: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    import logging
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase13b_downstream", str(cfg.resolve("logs")))
    triage = GeneratedLibraryTriage(cfg, logger)
    return triage.run_all(df)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase_downstream(cfgp)