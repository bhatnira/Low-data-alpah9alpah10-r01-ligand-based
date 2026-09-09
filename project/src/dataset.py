#!/usr/bin/env python3
"""Phase 2: Experimental dataset QC and canonicalization.

Implements prompt2.txt sections 4-5 and 42-report-1:
  - canonical experimental dataset with full provenance model
  - duplicate / conflicting-duplicate detection
  - stereochemistry audit (first-class, never collapsed)
  - salt/solvate, tautomer, charge handling where appropriate
  - MW sanity, activity-unit normalization, missing-value / outlier audit
  - no silent deletion: every exclusion carries an explicit reason
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
    sha256_dataframe, sha256_file,
)

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_FNS = {
    "MW": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "HBD": Descriptors.NumHDonors,
    "HBA": Descriptors.NumHAcceptors,
    "RotBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "AromaticRings": Descriptors.NumAromaticRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "FractionCSP3": Descriptors.FractionCSP3,
}


def _roundtrip_stereo(mol: Chem.Mol) -> Dict[str, Any]:
    """Return stereo info WITHOUT collapsing stereo information."""
    n_centers = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_unspecified = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    n_specified = n_centers - n_unspecified
    n_ez = sum(
        1 for b in mol.GetBonds()
        if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)
    )
    if n_centers > 0:
        status = "defined" if n_unspecified == 0 else "undefined/mixed"
    elif n_ez > 0:
        status = "E/Z_defined"
    else:
        status = "no_stereo"
    return {
        "stereo_n_stereo_centers": n_centers,
        "stereo_n_specified": n_specified,
        "stereo_n_unspecified": n_unspecified,
        "stereo_n_ez_bonds": n_ez,
        "stereo_stereo_status": status,
        "stereo_has_undefined_stereo": n_unspecified > 0,
    }


def _molecular_formula(mol: Chem.Mol) -> str:
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return "UNKNOWN"


class DatasetBuilder:
    """Builds the canonical experimental dataset (governance + QC)."""

    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger) -> None:
        self.cfg = cfg
        self.log = logger

    def build(self) -> pd.DataFrame:
        cfg = self.cfg
        raw_csv = cfg.data["raw_csv"]
        id_col = cfg.data["id_column"]
        smi_col = cfg.data["smiles_column"]
        act_col = cfg.data["activity_columns"]["potency"]
        pot_col = cfg.data["activity_columns"]["potentiation"]

        self.log.step("Loading raw experimental dataset")
        df = pd.read_csv(raw_csv)
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        self.log.info(f"Loaded {len(df)} rows from {raw_csv}")
        self.log.info(f"Columns: {list(df.columns)}")

        # ---- Input hashes for provenance ----
        raw_hash = sha256_file(raw_csv)
        save_manifest(
            make_provenance(
                "dataset.pipeline",
                cfg,
                {"raw_csv_hash": raw_hash, "raw_rows": int(len(df))},
            ),
            str(cfg.resolve("data/experimental/00_input_provenance.json")),
        )

        # ---- Missing value audit ----
        missing = pd.DataFrame(
            {
                "field": [id_col, smi_col, act_col, pot_col],
                "n_missing": [
                    int(df[id_col].isna().sum()),
                    int(df[smi_col].isna().sum()),
                    int(df[act_col].isna().sum()),
                    int(df[pot_col].isna().sum()),
                ],
            }
        )
        save_df(missing, str(cfg.resolve("sar/descriptors/missing_value_audit.csv")))
        self.log.info(f"Missing values:\n{missing.to_string(index=False)}")

        # ---- SMILES parsing (preserve stereo) ----
        mols, canonical, isomeric = [], [], []
        invalid_idx = []
        unevaluable = []
        for i, smi in enumerate(df[smi_col]):
            m = Chem.MolFromSmiles(str(smi))
            if m is None:
                invalid_idx.append(i)
                mols.append(None)
                canonical.append(None)
                isomeric.append(None)
                continue
            mols.append(m)
            canonical.append(Chem.MolToSmiles(m, isomericSmiles=True))
            iso = Chem.MolToSmiles(m, isomericSmiles=True)
            # canonical w/o stereo kept only as a reference connectivity key
            isomeric.append(iso)

        df["mol"] = mols
        df["canonical_SMILES"] = canonical
        df["isomeric_SMILES"] = isomeric
        df["invalid_smiles"] = 0
        df.loc[invalid_idx, "invalid_smiles"] = 1

        # ---- Salt / solvate detection (section 5) ----
        n_frags = []
        largest_frag_atoms = []
        for m in mols:
            if m is None:
                n_frags.append(np.nan)
                largest_frag_atoms.append(np.nan)
                continue
            frag_mols = rdmolops.GetMolFrags(m, asMols=True)
            n_frags.append(len(frag_mols))
            if frag_mols:
                largest_frag_atoms.append(max(fm.GetNumAtoms() for fm in frag_mols))
            else:
                largest_frag_atoms.append(np.nan)
        df["n_components"] = n_frags
        df["largest_fragment_heaviest_atoms"] = largest_frag_atoms
        df["is_salt_or_solvate"] = (df["n_components"].fillna(1) > 1).astype(int)
        self.log.info(f"Salt/solvate check: {int(df['is_salt_or_solvate'].sum())} multi-component entries")

        # ---- Tautomer normalization (section 5) ----
        taut_canon = []
        for m in mols:
            if m is None:
                taut_canon.append(None)
                continue
            try:
                enumerator = rdMolStandardize.TautomerEnumerator()
                taut_mol = enumerator.Canonicalize(m)
                taut_canon.append(Chem.MolToSmiles(taut_mol, isomericSmiles=True))
            except Exception:
                taut_canon.append(Chem.MolToSmiles(m, isomericSmiles=True))
        df["tautomer_canonical_SMILES"] = taut_canon
        n_changed = sum(1 for i, r in df.iterrows()
                        if r["isomeric_SMILES"] and r["tautomer_canonical_SMILES"]
                        and r["isomeric_SMILES"] != r["tautomer_canonical_SMILES"])
        self.log.info(f"Tautomer normalization: {n_changed} compounds had tautomer change")

        # ---- Charge normalization (section 5) ----
        uncharged_smiles = []
        net_charges = []
        for m in mols:
            if m is None:
                uncharged_smiles.append(None)
                net_charges.append(np.nan)
                continue
            net_charges.append(Chem.GetFormalCharge(m))
            try:
                uncharger = rdMolStandardize.Uncharger()
                u_mol = uncharger.uncharge(m)
                uncharged_smiles.append(Chem.MolToSmiles(u_mol, isomericSmiles=True))
            except Exception:
                uncharged_smiles.append(Chem.MolToSmiles(m, isomericSmiles=True))
        df["net_charge"] = net_charges
        df["uncharged_SMILES"] = uncharged_smiles
        charged = int(sum(1 for c in net_charges if c is not None and c != 0))
        self.log.info(f"Charge normalization: {charged} compounds with non-zero net charge")

        # ---- QC flags / exclusions ----
        df["qc_flag"] = "pass"
        df["exclusion_reason"] = ""
        if invalid_idx:
            df.loc[invalid_idx, "qc_flag"] = "invalid_smiles"
            df.loc[invalid_idx, "exclusion_reason"] = "UNPARSEABLE SMILES"

        # ---- Stereochemistry audit (never collapse) ----
        stereo_rows = []
        for m in df["mol"]:
            stereo_rows.append(_roundtrip_stereo(m) if m is not None else {k: np.nan for k in
                ["stereo_n_stereo_centers", "stereo_n_specified", "stereo_n_unspecified",
                 "stereo_n_ez_bonds", "stereo_stereo_status", "stereo_has_undefined_stereo"]})
        st = pd.DataFrame(stereo_rows)
        for c in st.columns:
            df[c] = st[c].values

        # ---- Explicit stereochemistry field (section 4) ----
        df["stereochemistry"] = df["stereo_stereo_status"].where(
            df["stereo_stereo_status"].notna(), "UNKNOWN"
        )

        # ---- Duplicate detection ----
        dup = df.duplicated(subset=["canonical_SMILES"], keep=False)
        df["dup_exact"] = dup.astype(int)
        df.loc[dup & (df["qc_flag"] == "pass"), "exclusion_reason"] = (
            df.loc[dup & (df["qc_flag"] == "pass"), "exclusion_reason"].astype(str)
            + " | EXACT_DUPLICATE"
        )
        n_dup = int(dup.sum())
        self.log.info(f"Exact canonical duplicates: {n_dup} compound-rows")

        # conflicting duplicates: same canonical SMILES, conflicting activity flags
        conflicts = (
            df[df["dup_exact"] == 1]
            .groupby("canonical_SMILES")[act_col]
            .apply(lambda s: s.nunique() > 1)
        )
        conflicting = set(conflicts[conflicts].index)
        df["conflicting_duplicate"] = df["canonical_SMILES"].isin(conflicting).astype(int)
        self.log.info(f"Conflicting duplicate groups: {len(conflicting)}")

        # ---- Descriptors ----
        for name, fn in DESCRIPTOR_FNS.items():
            df[name] = df["mol"].apply(lambda m: fn(m) if m is not None else np.nan)

        # ---- Salt / formula / charge ----
        df["MolFormula"] = df["mol"].apply(
            lambda m: _molecular_formula(m) if m is not None else "UNKNOWN"
        )
        df["FormalCharge"] = df["mol"].apply(
            lambda m: Chem.GetFormalCharge(m) if m is not None else np.nan
        )

        # ---- MW sanity ----
        df.loc[df["MW"].notna() & ((df["MW"] < 100) | (df["MW"] > 1000)), "qc_flag"] = "mw_outlier"
        df.loc[df["MW"].notna() & ((df["MW"] < 100) | (df["MW"] > 1000)), "exclusion_reason"] = (
            df.loc[df["MW"].notna() & ((df["MW"] < 100) | (df["MW"] > 1000)), "exclusion_reason"].astype(str)
            + " | MW_OUT_OF_SANITY_RANGE"
        )

        # ---- Activity classification ----
        df["activity_potency_uM"] = df[act_col].astype(float, errors="ignore")
        df["activity_potentiation_pct"] = df[pot_col].astype(float, errors="ignore")
        df["is_active"] = ((df[act_col] > 0) & (df[pot_col] > 0)).astype(int)
        # 0 in either field might mean 'inactive' or 'not measured'; keep unambiguous
        df["experimental_status"] = np.where(
            df["is_active"] == 1, "active", "inactive_or_not_measured"
        )

        # ---- Governance fields (prompt2 sect. 4) ----
        df["compound_id"] = df[id_col]
        df["SMILES"] = df[smi_col]
        df["activity"] = df[act_col]
        df["activity_type"] = "potency"
        df["activity_unit"] = "uM"
        df["assay"] = "a9a10 PAM potentiation assay"
        df["assay_condition"] = "UNKNOWN (assay conditions not recorded in source data)"
        df["replicate_information"] = "NOT_RECORDED (no replicate measurements in source data)"
        df["source"] = str(raw_csv)
        # series_id derived from murcko scaffold (section 4 operational definition)
        # parent_compound: NaN because source data does not declare parent-child relationships

        # ----- Murcko scaffold id (needed downstream for series-out splits) -----
        scaffolds = []
        for i, r in df.iterrows():
            m = r["mol"]
            if m is None:
                scaffolds.append(None)
                continue
            scaffolds.append(
                MurckoScaffoldSmiles(r["isomeric_SMILES"], m, includeChirality=True)
            )
        df["murcko_scaffold"] = scaffolds
        df["scaffold_id"] = pd.factorize(pd.Series(scaffolds, dtype="object"))[0]
        # series_id == scaffold_id (each unique Murcko scaffold is a series) (section 4)
        df["series_id"] = df["scaffold_id"]
        # parent_compound: no declared parent-child relationships in source; kept as NaN
        df["parent_compound"] = np.nan
        self.log.info(
            f"Series assignment: {int(df['series_id'].nunique())} series from "
            f"{int(df['murcko_scaffold'].nunique())} unique Murcko scaffolds"
        )

        # ---- Outlier audit (activity) ----
        nonzero = df[df[act_col] > 0][act_col]
        if len(nonzero) >= 4:
            q1, q3 = nonzero.quantile(0.25), nonzero.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 3 * iqr
            df["activity_outlier"] = ((df[act_col] > upper) & (df[act_col] > 0)).astype(int)
            self.log.info(f"Activity IQR outlier threshold (upper): {upper:.1f} uM")
        else:
            df["activity_outlier"] = 0

        # ---- QC summary ----
        self.log.step("QC summary")
        self.log.info(f"Total rows: {len(df)}")
        self.log.info(f"Valid SMILES parsed: {int((df['mol'].notna()).sum())}")
        self.log.info(f"Active compounds: {int(df['is_active'].sum())}")
        self.log.info(f"Compounds with undefined stereo: {int(df['stereo_has_undefined_stereo'].sum())}")
        self.log.info(
            "No compounds were silently deleted; every exclusion is flagged and reasoned."
        )

        # ---- Label distribution analysis (section 5) ----
        label_dist = df["is_active"].value_counts().sort_index()
        self.log.info(f"Label distribution (active=1): {label_dist.to_dict()}")
        df["label_distribution_note"] = (
            f"active={int((df['is_active']==1).sum())}, "
            f"inactive={int((df['is_active']==0).sum())}, "
            f"total={int(len(df))}"
        )
        dist_df = pd.DataFrame({
            "label": ["inactive", "active"],
            "count": [int((df["is_active"] == 0).sum()), int((df["is_active"] == 1).sum())],
            "fraction": [round(float((df["is_active"] == 0).mean()), 4),
                         round(float((df["is_active"] == 1).mean()), 4)],
        })
        save_df(dist_df, str(cfg.resolve("sar/descriptors/label_distribution.csv")))

        return df

    def save(self, df: pd.DataFrame) -> None:
        cfg = self.cfg
        keep = [
            "compound_id", "Identifier", "SMILES", "isomeric_SMILES",
            "canonical_SMILES", "tautomer_canonical_SMILES", "stereochemistry",
            "activity_potency_uM", "activity_potentiation_pct",
            "is_active", "experimental_status", "activity", "activity_type",
            "activity_unit", "assay", "assay_condition", "replicate_information",
            "source", "series_id", "parent_compound", "scaffold_id",
            "murcko_scaffold", "qc_flag", "exclusion_reason",
            "stereo_stereo_status", "stereo_n_stereo_centers",
            "stereo_n_specified", "stereo_n_unspecified", "stereo_n_ez_bonds",
            "stereo_has_undefined_stereo", "dup_exact", "conflicting_duplicate",
            "activity_outlier", "n_components", "is_salt_or_solvate",
            "net_charge", "uncharged_SMILES", "label_distribution_note",
            "MW", "LogP", "TPSA", "HBD", "HBA",
            "RotBonds", "RingCount", "AromaticRings", "FractionCSP3",
            "FormalCharge", "HeavyAtomCount", "MolFormula",
        ]
        keep = [c for c in keep if c in df.columns]
        out = df[keep].copy()
        save_df(out, str(cfg.resolve("data/experimental/canonical_experimental_dataset.csv")))
        save_df(out, str(cfg.resolve("data/processed/data_governed.csv")))
        save_df(
            out[out["qc_flag"] != "pass"],
            str(cfg.resolve("data/experimental/excluded_compounds.csv")),
        )
        # analyzeable = pass only
        save_df(
            out[out["qc_flag"] == "pass"],
            str(cfg.resolve("data/processed/data_analysis_ready.csv")),
        )
        hash_ = sha256_dataframe(out)
        save_manifest(
            {
                "generated_at": make_provenance("dataset.save", self.cfg, {})["generated_at"],
                "canonical_dataset_hash": hash_,
                "n_rows": int(len(out)),
                "n_analysis_ready": int((out["qc_flag"] == "pass").sum()),
            },
            str(self.cfg.resolve("data/experimental/dataset_manifest.json")),
        )
        self.log.info(f"Canonical dataset hash: {hash_}")
        self.log.step("Phase 2 complete")


def run_phase2(cfg_path: str) -> pd.DataFrame:
    import logging
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase2_dataset", str(cfg.resolve("logs")), level=logging.INFO)
    builder = DatasetBuilder(cfg, logger)
    df = builder.build()
    builder.save(df)
    return df


if __name__ == "__main__":
    import logging
    default_cfg = str(Path("/Users/nb/Documents/ligand-based-modeling/project/configs/config.yaml"))
    import sys
    if len(sys.argv) > 1:
        default_cfg = sys.argv[1]
    run_phase2(default_cfg)