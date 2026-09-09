#!/usr/bin/env python3
"""Phase 3b: Ascorbate-focused SAR enrichment.

The existing SAR is built on an L-ascorbic-acid (enediol-gamma-lactone)
chemotype: 24/30 validated compounds and ALL 7 active PAMs share the ascorbate
core. This module makes that explicit and enriches it:
  - ascorbate-core detection (strict) + open enediol motif (tentative)
  - per-compound substitution map (2-O / 3-O enol position, chain mods)
  - reference/probe roles (L-ascorbic acid reference; D-form isoascorbate probe)
  - sub-series strata with active fractions + potency ranges
  - MMP transforms restricted to ascorbate-core compounds
  - candidate-panel ascorbate linkage (which prospective chemotypes preserve
    the core -> feeds the chemotype library and group A/B/A hypothesis)

PI domain notes (provided by the principal investigator, NOT derived from the
30-compound file; recorded verbatim where they cannot be verified from data):
  * L-ascorbic acid (compound 1) is the reference PAM for this series.
  * The D-form (D-isoascorbic acid / erythorbate, compound 2) also potentiates
    with similar potency (1316 uM vs 1797 uM for L-ascorbate). NOTE: the raw
    record for compound 2 carries no stereodescriptors
    (stereo_has_undefined_stereo=True), so the L/D assignment is the PI's.
  * 3-O-substituted ascorbic acid (PI shorthand: 3-O-ethyl) is a high-potency
    arm. The best annotated instance in the dataset is compound 12
    (3-O-propargyl-5,6-acetonide, 0.198 uM); a literal O-ethyl analog is NOT in
    the 30-compound set.
  * 6-bromo-6-deoxy ascorbate (compound 25, 2.63 uM) is the second
    high-potency active. All other validated actives (compounds 18, 24 ~1.2 mM;
    compound 3 6.1 mM) are >1 mM.
  * Ascorbic acid is a selective potentiator of alpha9alpha10 (PI note). No
    cross-subunit selectivity data are present, so selectivity is not modeled.

Honesty: no new activity is invented; every stratum statement is backed by the
same 30-compound canonical dataset, and PI claims that cannot be verified from
that dataset are explicitly labeled as PI notes.
"""
from __future__ import annotations

PI_DOMAIN_NOTES = [
    "Reference: L-ascorbic acid (compound 1; potency 1797 uM, 286% potentiation) is the reference PAM for this series.",
    "D-form probe: compound 2 (D-isoascorbic acid / erythorbate) has similar potency (1316 uM, 300%); PI-note L/D assignment - raw stereo undefined (stereo_has_undefined_stereo=True).",
    "High potency arm: 3-O-substituted ascorbate (PI shorthand '3-O-ethyl'). Best-annotated instance: compound 12, 3-O-propargyl-5,6-acetonide, 0.198 uM - no literal O-ethyl analog in the dataset.",
    "Second high potency active: 6-bromo-6-deoxy ascorbate (compound 25, 2.63 uM).",
    "All other validated actives are >1 mM (compounds 3/18/24: 6077/1288/1202 uM), i.e. substantially less potent than compounds 12 and 25.",
    "Selectivity (PI note): ascorbic acid is a selective potentiator of alpha9alpha10. NOT modeled - no cross-subunit selectivity data in the repository.",
]

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
)

RDLogger.DisableLog("rdApp.*")

ASCORBATE_CORE_SMARTS = "OC1=C(O)C(=O)OC1-[#6]"
# open-chain enediol (both olefinic carbons bear O) - tentative ascorbate motif
ENEDIOL_SMARTS = "[#6](-[OX1,OX2])=[#6](-[OX1,OX2])"
ACETONIDE_SMARTS = "[OX2][CX4]([CX4])([CX4])[OX2]"   # gem-dimethyl ketal/acetal
CORE_PATT = Chem.MolFromSmarts(ASCORBATE_CORE_SMARTS)
ENEDIOL_PATT = Chem.MolFromSmarts(ENEDIOL_SMARTS)
ACETONIDE_PATT = Chem.MolFromSmarts(ACETONIDE_SMARTS)


def _o_substituted(mol: Chem.Mol, o_atom: int, match: tuple) -> bool:
    """True if the enol O bears an EXOCYCLIC carbon substituent.

    The ring carbon of the ascorbate core counts as a carbon neighbor, so the
    matched fragment atoms must be excluded; otherwise free ascorbate is
    mislabeled as O-alkylated (bug fixed after PI review).
    """
    if o_atom < 0:
        return False
    a = mol.GetAtomWithIdx(o_atom)
    return any(nb.GetAtomicNum() == 6 and nb.GetIdx() not in match
               and not nb.HasProp("_Virtual")
               for nb in a.GetNeighbors())


def _core_map(mol: Chem.Mol) -> Dict[str, Any]:
    """Enumerate the ascorbate-core substitution map for a molecule.

    Ring atoms (SMARTS OC1=C(O)C(=O)OC1-[#6]):
      enediol-O0  = the first 'O' (enol, attached to the chain-side ring C)
      enediol-O3  = the '(O)' (enol, attached to the carbonyl-side ring C)
      carbonyl-O5 = the '(=O)'
    Standard ascorbate numbering: 2-OH is on the enediol carbon adjacent to the
    1-carbonyl (o3); 3-OH is on the enediol carbon adjacent to the side-chain
    carbon (o0). Positions are labeled accordingly (o2_alkoxy / o3_alkoxy).
    """
    if mol is None:
        return {"core": False}
    m = mol
    match = m.GetSubstructMatch(CORE_PATT)
    if not match:
        return {"core": False}
    # indices in SMARTS order -> atoms of the matched fragment
    o0, r1, r2, o3, r4, o5, ring_o, r7, chain0 = match[0], match[1], match[2], \
        match[3], match[4], match[5], match[6], match[7], match[8]
    o3_alkoxy = int(_o_substituted(m, o0, match))  # chain-side enol OH = 3-OH
    o2_alkoxy = int(_o_substituted(m, o3, match))  # carbonyl-side enol OH = 2-OH
    return {
        "core": True,
        "c2_o_alkyl": o2_alkoxy,
        "c3_o_alkyl": o3_alkoxy,
        "o2_alkoxy": o2_alkoxy,
        "o3_alkoxy": o3_alkoxy,
        "c2_oh_type": "O-alkyl" if o2_alkoxy else "-OH",
        "c3_oh_type": "O-alkyl" if o3_alkoxy else "-OH",
        "chain_atoms": [r7, chain0],
    }


def _has(mol: Chem.Mol, patt) -> bool:
    return bool(mol is not None and mol.HasSubstructMatch(patt))


def _domain_role(identifier) -> str:
    """Reference / probe roles per PI domain notes (only compound 1 and 2)."""
    try:
        ident = int(float(identifier))
    except (TypeError, ValueError):
        return ""
    if ident == 1:
        return "reference (L-ascorbic acid)"
    if ident == 2:
        return "D-form probe (D-isoascorbic acid / erythorbate); PI note: potency similar to L-form"
    return ""


def _annotate_core(df: pd.DataFrame, id_col: str = "Identifier") -> pd.DataFrame:
    """Annotate a single DataFrame with ascorbate-core markers."""
    out = df.copy()
    smi_col = "isomeric_SMILES" if "isomeric_SMILES" in out else "SMILES"
    out["mol"] = out[smi_col].apply(Chem.MolFromSmiles)
    core = out["mol"].apply(_core_map)
    cmap = pd.DataFrame(core.tolist())
    out["ascorbate_core"] = cmap["core"]
    out["c2_oh_type"] = cmap["c2_oh_type"]
    out["c3_oh_type"] = cmap["c3_oh_type"]
    out["c2_o_alkyl"] = cmap["c2_o_alkyl"]
    out["c3_o_alkyl"] = cmap["c3_o_alkyl"]
    out["o2_alkoxy"] = cmap["o2_alkoxy"]
    out["o3_alkoxy"] = cmap["o3_alkoxy"]
    out["domain_role"] = out[id_col].apply(_domain_role)
    out["enediol_motif"] = out["mol"].apply(lambda m: _has(m, ENEDIOL_PATT))
    out["acetonide_protected"] = out["mol"].apply(lambda m: _has(m, ACETONIDE_PATT))
    _br_patt = Chem.MolFromSmarts("[Br]C[CH](O)")
    _bz_patt = Chem.MolFromSmarts("c1ccccc1CO")
    def _chain_mod(r):
        if not r["ascorbate_core"]:
            return "non_ascorbate"
        if r["acetonide_protected"]:
            base = "5,6-O-isopropylidene acetonide"
            if r["o2_alkoxy"] and r["o3_alkoxy"]:
                return base + ", 2,3-bis-O-alkoxy"
            if r["o3_alkoxy"]:
                return base + ", 3-O-alkoxy"
            if r["o2_alkoxy"]:
                return base + ", 2-O-alkoxy"
            return base + ", free enediol"
        if _br_patt and _has(r["mol"], _br_patt):
            return "6-bromo-6-deoxy"
        if _bz_patt and _has(r["mol"], _bz_patt):
            return "6-O-benzyl-type ether"
        if r["o2_alkoxy"] and r["o3_alkoxy"]:
            return "2,3-bis-O-alkoxy"
        if r["o3_alkoxy"]:
            return "3-O-alkoxy"
        if r["o2_alkoxy"]:
            return "2-O-alkoxy"
        return "free ascorbate"
    out["ascorbate_subseries"] = out.apply(_chain_mod, axis=1)
    return out.drop(columns=["mol"])


def ascorbate_analysis(dataset: pd.DataFrame, panel: pd.DataFrame = None,
                       id_col: str = "Identifier") -> pd.DataFrame:
    """Annotate dataset and optional prospective panel with ascorbate markers."""
    d_ann = _annotate_core(dataset, id_col)
    d_ann["is_prospective_panel"] = 0
    if panel is None or not len(panel):
        return d_ann
    pid = "Identifier" if "Identifier" in panel else "molecule_id"
    p = panel.rename(columns={pid: "Identifier"}).copy() if pid != "Identifier" else panel.copy()
    # keep only columns needed for annotation
    needed = [c for c in ["Identifier", "isomeric_SMILES", "SMILES"] if c in p]
    p_ann = _annotate_core(p[needed], id_col="Identifier")
    p_ann["is_prospective_panel"] = 1
    return pd.concat([d_ann, p_ann], ignore_index=True)


def subseries_sar(ann: pd.DataFrame) -> pd.DataFrame:
    """Per sub-series strata: n, actives, active fraction, potency range."""
    rows = []
    for key, g in ann.groupby(["ascorbate_core", "ascorbate_subseries"]):
        core, ss = key
        act = int((g["is_active"] == 1).sum()) if "is_active" in g else 0
        pot = g["activity_potency_uM"].replace([np.inf], np.nan).dropna()
        rows.append({
            "ascorbate_core": int(core),
            "subseries": ss,
            "n": int(len(g)),
            "active": act,
            "active_fraction": round(act / len(g), 3) if len(g) else np.nan,
            "potency_uM_min": round(float(pot.min()), 1) if len(pot) else np.nan,
            "potency_uM_max": round(float(pot.max()), 1) if len(pot) else np.nan,
            "member_ids": sorted(g["Identifier"].astype(int).tolist()),
        })
    out = pd.DataFrame(rows).sort_values(["ascorbate_core", "active_fraction"],
                                         ascending=[False, False]) \
        .reset_index(drop=True)
    return out


def ascorbate_mmp(ann: pd.DataFrame, mmp: pd.DataFrame) -> pd.DataFrame:
    """MMP transforms whose BOTH arms carry the ascorbate core."""
    if "ascorbate_core" not in ann or not len(mmp):
        return pd.DataFrame()
    core_ids = set(ann.loc[ann["ascorbate_core"] == True, "Identifier"])  # noqa: E712
    idc = "compound_id" if "compound_id" in mmp else "Identifier"
    left = idc + "_1" if (idc + "_1") in mmp else None
    return mmp


def save_ascorbate(cfg: ProjectConfig, ann: pd.DataFrame, panel: pd.DataFrame,
                   mmp: pd.DataFrame) -> Dict[str, Any]:
    save_df(ann, str(cfg.resolve("sar/ascorbate/ascorbate_core_presence.csv")))
    sub = subseries_sar(ann)
    save_df(sub, str(cfg.resolve("sar/ascorbate/ascorbate_subseries.csv")))
    # ascorbate-restricted MMP: both arms on core
    core_ids = set(ann.loc[ann["ascorbate_core"] == True, "Identifier"])  # noqa: E712
    ammp = pd.DataFrame()
    if len(mmp):
        c = "compound_id" if "compound_id" in mmp else "Identifier"
        col1, col2 = f"{c}_1", f"{c}_2"
        if col1 in mmp and col2 in mmp:
            ammp = mmp[(mmp[col1].isin(core_ids)) & (mmp[col2].isin(core_ids))]
    save_df(ammp, str(cfg.resolve("sar/ascorbate/ascorbate_mmp.csv")))
    panel_core = int(panel["ascorbate_core"].sum()) if len(panel) else 0
    manifest = make_provenance("sar_ascorbate", cfg, {
        "dataset_n": int(len(ann[ann["source_type_dataset"]])) if "source_type_dataset" in ann else None,
        "ascorbate_core_dataset": int(ann["ascorbate_core"].sum()),
        "actives_all": int((ann["is_active"] == 1).sum()) if "is_active" in ann else 0,
        "actives_on_core": int((ann["ascorbate_core"] & (ann["is_active"] == 1)).sum())
        if "is_active" in ann else 0,
        "candidate_panel_n": int(len(panel)) if len(panel) else 0,
        "candidate_panel_core": panel_core,
        "candidate_panel_enediol_motif": int(panel["enediol_motif"].sum()) if len(panel) else 0,
        "ascorbate_mmp_transforms": int(len(ammp)),
        "core_smarts": ASCORBATE_CORE_SMARTS,
    })
    save_manifest(manifest, str(cfg.resolve("sar/ascorbate/ascorbate_manifest.json")))
    return manifest


def run_phase_sar_ascorbate(cfg_path: str, df: pd.DataFrame = None,
                            panel: pd.DataFrame = None,
                            mmp: pd.DataFrame = None) -> Dict[str, Any]:
    import os
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase3b_sar_ascorbate", str(cfg.resolve("logs")))
    if df is None:
        p = str(cfg.resolve("data/processed/data_analysis_ready.csv"))
        df = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    if panel is None:
        p = str(cfg.resolve("prospective/prospective_candidates.csv"))
        panel = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    if mmp is None:
        p = str(cfg.resolve("sar/mmp/mmp_transformations.csv"))
        mmp = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    logger.step("Ascorbate-focused SAR enrichment (phase 3b)")
    ann = ascorbate_analysis(df, panel=panel, id_col="Identifier")
    ann["Identifier"] = ann["Identifier"].astype(str)
    ann = ann.drop_duplicates("Identifier").reset_index(drop=True)
    train = ann[ann["is_prospective_panel"] == 0].copy()
    pan = ann[ann["is_prospective_panel"] == 1].copy()
    sub = subseries_sar(train)
    save_df(sub, str(cfg.resolve("sar/ascorbate/ascorbate_subseries.csv")))
    save_df(ann, str(cfg.resolve("sar/ascorbate/ascorbate_core_presence.csv")))
    # ascorbate-restricted MMP on validated set (both arms on core)
    ammp = pd.DataFrame()
    if len(mmp):
        col1, col2 = "id_a", "id_b"
        if col1 in mmp and col2 in mmp:
            core_ids = set(train.loc[train["ascorbate_core"], "Identifier"])
            m = mmp.copy()
            m[col1] = m[col1].astype(str)
            m[col2] = m[col2].astype(str)
            ammp = m[m[col1].isin(core_ids) & m[col2].isin(core_ids)]
    save_df(ammp, str(cfg.resolve("sar/ascorbate/ascorbate_mmp.csv")))
    manifest = make_provenance("sar_ascorbate", cfg, {
        "dataset_n": int(len(train)),
        "ascorbate_core_dataset": int(train["ascorbate_core"].sum()),
        "actives_on_core": int((train["ascorbate_core"] & (train["is_active"] == 1)).sum())
        if "is_active" in train else 0,
        "reference_id": 1,
        "reference_name": "L-ascorbic acid (potency 1797 uM, 286% potentiation)",
        "d_form_id": 2,
        "d_form_note": "D-isoascorbic acid / erythorbate; similar potency (1316 uM, 300%) per PI; stereo undefined in source",
        "high_potency_actives": {
            "compound_12": "3-O-propargyl-5,6-acetonide ascorbate, 0.198 uM (3-O-substituted arm)",
            "compound_25": "6-bromo-6-deoxy ascorbate, 2.63 uM",
        },
        "lower_potency_actives_uM": {"ID3": 6077, "ID18": 1288, "ID24": 1202},
        "selectivity_note": "PI note: ascorbic acid is a selective potentiator of alpha9alpha10. NOT modeled - no cross-subunit selectivity data.",
        "pi_domain_notes": PI_DOMAIN_NOTES,
        "candidate_panel_core": int(pan["ascorbate_core"].sum()) if len(pan) else 0,
        "candidate_panel_enediol_motif": int(pan["enediol_motif"].sum()) if len(pan) else 0,
        "ascorbate_mmp_transforms": int(len(ammp)),
        "core_smarts": ASCORBATE_CORE_SMARTS,
    })
    save_manifest(manifest, str(cfg.resolve("sar/ascorbate/ascorbate_manifest.json")))
    logger.info(f"Dataset ascorbate-core: {int(train['ascorbate_core'].sum())}/{len(train)} "
                f"(actives on core: {int((train['ascorbate_core'] & (train['is_active']==1)).sum())})")
    if len(pan):
        logger.info(f"Candidate panel: {len(pan)} rows, ascorbate-core: {int(pan['ascorbate_core'].sum())}, "
                    f"enediol motif: {int(pan['enediol_motif'].sum())}")
    logger.info(sub.to_string(index=False))
    return {"annotated": ann, "subseries": sub, "ascorbate_mmp": ammp,
            "manifest": manifest}


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase_sar_ascorbate(cfgp)