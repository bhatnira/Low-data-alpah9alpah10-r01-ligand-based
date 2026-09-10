#!/usr/bin/env python3
"""Selectivity panel: curation + chemotype scan + model benchmark for the
non-alpha9alpha10 nAChR modulator set in <repo>/otherModulators.csv.

Purpose (per project design request):
  These are modulators of OTHER nAChR subtypes (mostly alpha7 and alpha4beta2
  PAMs, the CMPI/isoxazole-indole series, and nAChR-active flavonoids). They
  are NOT alpha9alpha10 measurements, so they are deliberately kept SEPARATE
  from the alpha9alpha10 training set (data integrity). Instead they form a
  documented "selectivity / decoys" panel used to:

    1. Curation    - canonicalize + annotate (descriptors, murcko scaffolds,
                     subtype/class) with full provenance and QA flags.
    2. Chemotype   - Bemis-Murcko scaffold scan vs the alpha9alpha10 ascorbate
                     training library and the candidate chemotype families.
    3. Benchmark   - score the panel with the frozen alpha9alpha10 PAM model to
                     characterize what the alpha9alpha10 model predicts for
                     known other-subtype modulators (the decoy baseline any new
                     designed hit should beat / discriminate against).

Outputs land under <project>/selectivity/:
  selectivity_panel.csv            curated panel
  selectivity_physchem_summary.csv descriptor coverage summary
  selectivity_scaffold_scan.csv    Bemis-Murcko overlap vs a9a10 library + families
  selectivity_model_scores.csv     frozen-model predictions on the panel
  selectivity_report.md            narrative report
  selectivity_manifest.json        provenance
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
    sha256_dataframe, sha256_file,
)
from src.dataset import DESCRIPTOR_FNS

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# subtype / chemotype annotation of the panel (literature-informed labels)
# ---------------------------------------------------------------------------
# classLabel in the source file encodes the source's own binary class.  We
# additionally annotate a best-effort subtype/chemotype note so downstream use
# is interpretable.  These notes are descriptive, not fabrications.
SUBTYPE_NOTES: Dict[str, str] = {
    # alpha4beta2 PAMs / alpha4beta2-active potentiators
    "NS-9283.mol": "alpha4beta2 PAM (3-(1,2,4-oxadiazolyl)-pyridine series)",
    "NS-206.mol": "alpha7 PAM (benzoxanthin/quinoxalinone NS series)",
    # isoxazole / pyrazole "CMPI" alpha7-type PAM series (Compound 9 / 1 / CMPI / 2* / 12*)
    "Compound9.mol": "isoxazole-piperazine PAM series (a7-type)",
    "CMPI.mol": "isoxazole-pyrazole CMPI PAM series (a7-type)",
    "1.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2a.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2b.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2c.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2d.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2e.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2f.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2g.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2h.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2i.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2j.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2k.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2l.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2o.mol": "isoxazole-piperazine PAM series (a7-type)",
    "2p.mol": "isoxazole-piperazine PAM series (a7-type)",
    "12a.mol": "isoxazole-piperazine PAM series (a7-type)",
    "12b.mol": "isoxazole-piperazine PAM series (a7-type)",
    "12c.mol": "isoxazole-piperazine PAM series (a7-type)",
    "12d.mol": "isoxazole-piperazine PAM series (a7-type)",
    "12e.mol": "isoxazole-piperazine PAM series (a7-type)",
    # substituted indole/azaindole series (DFBR / Com* / Struc*)
    "dfbr.mol": "indole PAM series (a7-type)",
    "Com8.mol": "indole PAM series (a7-type)",
    "Com9.mol": "indole PAM series (a7-type)",
    "Com10.mol": "indole PAM series (a7-type)",
    "Com11.mol": "azaindole PAM series (a7-type)",
    "Com12.mol": "indole PAM series (a7-type)",
    "Com13.mol": "indole PAM series (a7-type)",
    "Com14.mol": "indole PAM series (a7-type)",
    "Com16.mol": "indole PAM series (a7-type)",
    "Com17.mol": "indole PAM series (a7-type)",
    "Com18.mol": "indole PAM series (a7-type)",
    "Com19.mol": "indole PAM series (a7-type)",
    "Com20.mol": "indole PAM series (a7-type)",
    "Com21.mol": "indole PAM series (a7-type)",
    "Com22.mol": "indole PAM series (a7-type)",
    "Com23.mol": "indole PAM series (a7-type)",
    "Com24.mol": "indole PAM series (a7-type)",
    "Com25.mol": "indole PAM series (a7-type)",
    "Struc2.mol": "indole PAM series (a7-type)",
    "Struc3.mol": "indole PAM series (a7-type)",
    "Struc4.mol": "indole PAM series (a7-type)",
    "Struc5.mol": "indole PAM series (a7-type)",
    "Struc6.mol": "indole PAM series (a7-type)",
    "Struc7.mol": "indole PAM series (a7-type)",
    "Struc8.mol": "indole PAM series (a7-type)",
    "Struc9.mol": "indole PAM series (a7-type)",
    "Struc10.mol": "indole PAM series (a7-type)",
    "Struc11.mol": "indole PAM series (a7-type)",
    "Struc12.mol": "indole PAM series (a7-type)",
    # known alpha7 PAMs
    "NS-1738 (1).mol": "alpha7 PAM (thiophene-urea)",
    "PNU-120596.mol": "alpha7 PAM (classical Type II)",
    "PAM-2.mol": "alpha7 PAM (cinnamide NS-1738 analog)",
    "PAM-4.mol": "alpha7 PAM (cinnamide NS-1738 analog)",
    "TQS.mol": "alpha7 PAM (quinoxalin-2-one Type I/II)",
    "A-867744.mol": "alpha7 PAM (Type II)",
    "5-hydroxyindole (5-HI).mol": "alpha7 PAM (5-HI)",
    "4.mol": "alpha7 PAM (pyrazole-sulfonamide series)",
    "TBS-156.mol": "alpha7 PAM (triazine-sulfonamide series)",
    "TBS-516.mol": "alpha7 PAM (triazine-sulfonamide series)",
    "3.mol": "alpha7 PAM (thiophene ketoamide series)",
    "LL-00066471.mol": "alpha7 PAM (pyridinone series)",
    # nAChR-active flavonoids / natural products
    "Quercetin.mol": "flavonoid nAChR modulator (natural product)",
    "5,7-dihydroxy-4-phenylcoumarin.mol": "flavonoid nAChR modulator",
    "Isoliquirigenin.mol": "chalcone nAChR modulator (natural product)",
    "Curcumin.mol": "curcuminoid nAChR modulator (natural product)",
    "6.mol": "chalcone nAChR modulator (natural product)",
    "5.mol": "chalcone nAChR modulator (natural product)",
    "RGM079.mol": "flavonoid nAChR modulator",
    "Genistein.mol": "isoflavone nAChR modulator (natural product)",
    # others
    "HEPES.mol": "buffer/filter artifact (4-(2-hydroxyethyl)piperazine in assay)",
}

NATURAL_PRODUCTS = {"Quercetin.mol", "5,7-dihydroxy-4-phenylcoumarin.mol",
                    "Isoliquirigenin.mol", "Curcumin.mol", "6.mol", "5.mol",
                    "RGM079.mol", "Genistein.mol"}


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
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


def _murcko(mol: Chem.Mol) -> str:
    single = _largest_fragment(mol)
    if single is None:
        return ""
    try:
        s = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(single))
        if s:
            return s
    except Exception:
        pass
    return Chem.MolToSmiles(single, canonical=True)


def _stereo_summary(mol: Chem.Mol) -> Dict[str, Any]:
    if mol is None:
        return {"stereo_status": "UNKNOWN", "n_stereo_centers": np.nan,
                "n_undefined_stereo": np.nan}
    n = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    nu = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    if n == 0:
        status = "no_stereo"
    elif nu == 0:
        status = "defined"
    else:
        status = "undefined/mixed"
    return {"stereo_status": status, "n_stereo_centers": n,
            "n_undefined_stereo": nu}


PANEL_COLUMNS = ["File Name", "cleanedMol", "classLabel"]

# Directory repeated with later/expanded additions: drop a CSV here (same
# columns, or File Name + clearedMol + classLabel) and re-run the phase.  Each
# file is versioned in the manifest by content hash and merged by canonical
# SMILES (first-seen wins).
DEFAULT_EXPANSION_DIR = "data/selectivity/raw"


def _discover_panel_sources(cfg: ProjectConfig) -> List[str]:
    """Return the ordered, deduped list of source CSVs for the panel.

    Order (later files add to the panel; earlier files take precedence on
    duplicate canonical SMILES):
      1. <repo root>/otherModulators.csv        (original list)
      2. every *.csv in <project>/data/selectivity/raw/  (expansion drop-ins)
      3. config override: data.selectivity_panel.sources[].file
    """
    root = Path(cfg.project["root"])
    sources: List[str] = []

    legacy = root.parent / "otherModulators.csv"
    if legacy.exists():
        sources.append(str(legacy))

    expand_dir = root / DEFAULT_EXPANSION_DIR
    if expand_dir.exists():
        sources.extend(sorted(str(p) for p in expand_dir.glob("*.csv")))

    override = cfg.raw.get("data", {}).get("selectivity_panel", {}).get("sources", [])
    for entry in override:
        f = entry.get("file") if isinstance(entry, dict) else entry
        if f and str(f) not in sources:
            sources.append(str(f))
    return sources


class OtherModulatorPanel:
    """Curate + annotate the non-a9a10 nAChR modulator panel.

    Sources are discoverable and expandable (see _discover_panel_sources);
    the panel is -- by construction -- isolated from the alpha9alpha10
    training set (it is stored under <project>/selectivity and is never
    written into data/processed).  An explicit cross-contamination guard
    reports any exact-SMILES overlap with the alpha9alpha10 training set.
    """

    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger

    def curate(self, raw_csvs: List[str]) -> pd.DataFrame:
        self.log.step("Curate non-alpha9alpha10 nAChR selectivity panel")
        source_hashes = {p: sha256_file(p) for p in raw_csvs}

        rows = []
        for raw_csv in raw_csvs:
            try:
                raw = pd.read_csv(raw_csv)
            except Exception as e:
                self.log.warn(f"Could not read panel source {raw_csv}: {e}")
                continue
            raw.columns = [c.strip() if isinstance(c, str) else c for c in raw.columns]
            # drop trailing empty enum columns
            raw = raw.loc[:, [c for c in raw.columns if not c.startswith("Unnamed")]]
            missing = [c for c in PANEL_COLUMNS if c not in raw.columns]
            if missing:
                self.log.warn(f"Panel source {raw_csv} missing required columns "
                              f"{missing}; skipped")
                continue
            n_before = len(rows)
            for _, r in raw.iterrows():
                name = str(r.get("File Name", "")).strip()
                smi = str(r.get("cleanedMol", "")).strip()
                cl = r.get("classLabel")
                try:
                    class_label = float(cl)
                except (TypeError, ValueError):
                    class_label = np.nan
                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol is None:
                    rows.append({
                        "panel_id": name, "file_name": name, "source_label": class_label,
                        "source_file": raw_csv, "SMILES": smi, "canonical_SMILES": "",
                        "is_parseable": 0, "subtype_note": SUBTYPE_NOTES.get(name, ""),
                    })
                    continue
                canon = Chem.MolToSmiles(mol, isomericSmiles=True)
                st = _stereo_summary(mol)
                rows.append({
                    "panel_id": name, "file_name": name, "source_label": class_label,
                    "source_file": raw_csv, "SMILES": smi, "canonical_SMILES": canon,
                    "is_parseable": 1,
                    "murcko_scaffold": _murcko(mol),
                    "is_natural_product": int(name in NATURAL_PRODUCTS),
                    "subtype_note": SUBTYPE_NOTES.get(name, ""),
                    "stereo_status": st["stereo_status"],
                    "n_stereo_centers": st["n_stereo_centers"],
                    "n_undefined_stereo": st["n_undefined_stereo"],
                })
            self.log.info(f"Source {raw_csv}: +{len(rows) - n_before} rows")

        df = pd.DataFrame(rows)
        # descriptors (only on parseable molecules)
        df["mol"] = df["canonical_SMILES"].apply(
            lambda s: Chem.MolFromSmiles(s) if s else None)
        for dname, fn in DESCRIPTOR_FNS.items():
            df[dname] = df["mol"].apply(lambda m: fn(m) if m is not None else np.nan)
        df["n_heavy_atoms"] = df["mol"].apply(
            lambda m: m.GetNumHeavyAtoms() if m is not None else np.nan)

        # Lipinski/PAINS/BRENK drug-likeness (largest fragment) for a "decoys"
        # interpretation
        pains, brenk = _catalogs()
        def _dl(m):
            if m is None:
                return (np.nan, np.nan, np.nan)
            single = _largest_fragment(m)
            mw = Descriptors.MolWt(single)
            logp = Crippen.MolLogP(single)
            hbd = rdMolDescriptors.CalcNumHBD(single)
            hba = rdMolDescriptors.CalcNumHBA(single)
            viol = int((mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10))
            return (viol, int(pains.HasMatch(single)), int(brenk.HasMatch(single)))
        dl = df["mol"].apply(_dl)
        df["lipinski_violations"] = [x[0] for x in dl]
        df["pains_flag"] = [x[1] for x in dl]
        df["brenk_flag"] = [x[2] for x in dl]

        # QC: unparseable / fragment flags
        df["qc_flag"] = np.where(df["is_parseable"] == 1, "pass", "unparseable")

        # ---- dedupe by canonical SMILES (first-seen source wins) ----
        if "canonical_SMILES" in df.columns and df["canonical_SMILES"].notna().any():
            n_before = len(df)
            df = df.drop_duplicates(subset=["canonical_SMILES"], keep="first")
            self.n_deduped = int(n_before - len(df))
            self.log.info(f"Deduped {self.n_deduped} duplicate canonical SMILES; "
                          f"kept {len(df)} unique")
        else:
            self.n_deduped = 0

        # source attribution + hashes for provenance
        df["n_sources_merged"] = len(raw_csvs)
        self.source_hashes = source_hashes
        self.source_count = {os.path.basename(p): int((df["source_file"] == p).sum())
                             for p in raw_csvs}

        self.log.info(f"Panel rows: {len(df)}; parseable: {int(df['is_parseable'].sum())}; "
                      f"unparseable: {int((df['is_parseable']==0).sum())}")
        self.panel_hash = sha256_dataframe(
            df.drop(columns=["mol"], errors="ignore").fillna(""))
        self.raw_hash = "|".join(f"{os.path.basename(p)}={h}" for p, h in source_hashes.items())
        return df


def _catalogs():
    from rdkit.Chem import FilterCatalog
    def cat(kind):
        p = FilterCatalog.FilterCatalogParams()
        p.AddCatalog(kind)
        return FilterCatalog.FilterCatalog(p)
    return cat(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS), \
        cat(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)


def _morgan_array(mol: Chem.Mol, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _max_tanimoto(query_fp: np.ndarray, refs: np.ndarray) -> float:
    if refs.size == 0:
        return 0.0
    denom = np.maximum(
        (np.linalg.norm(query_fp) * np.linalg.norm(refs, axis=1)), 1e-12)
    sims = (query_fp @ refs.T) / denom
    return float(sims.max())


class SelectivityBenchmark:
    """Score the panel with the frozen a9a10 model + measure scaffold overlap."""

    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger,
                 model_artifact: str, panel: pd.DataFrame,
                 train: Optional[pd.DataFrame] = None,
                 families: Optional[pd.DataFrame] = None):
        self.cfg = cfg
        self.log = logger
        self.artifact_path = model_artifact
        self.panel = panel
        self.train = train
        self.families = families

    def run(self) -> Dict[str, pd.DataFrame]:
        import joblib
        art = joblib.load(self.artifact_path)
        clf = art["model"]
        scaler = art["scaler"]
        desc_cols = list(art.get("desc_columns", []) or [])
        pos = int(art.get("positive_class", 1))
        active_smiles = list(art.get("active_smiles", []))
        training_murcko = set(art.get("training_murcko", []))
        nbits = int(art.get("nbits", 2048))

        active_fps = np.vstack([_morgan_array(Chem.MolFromSmiles(s), nbits=nbits)
                                for s in active_smiles]) if active_smiles else \
            np.zeros((1, nbits))

        # scaffold overlap vs alpha9alpha10 training library
        train_scaffolds = set(self.train["murcko_scaffold"].astype(str)) if self.train is not None else set()
        fam_scaffolds = set(self.families["chemotype_scaffold"].astype(str)) \
            if self.families is not None else set()

        score_rows = []
        for _, r in self.panel.iterrows():
            s = r.get("canonical_SMILES", "")
            pred = np.nan
            uncertainty = np.nan
            max_sim_active = np.nan
            if s:
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    fp = _morgan_array(mol, nbits=nbits)
                    desc = [r.get(c, np.nan) if r.get(c, np.nan) == r.get(c, np.nan)
                            else 0.0 for c in desc_cols]
                    desc = [float(r[c]) if pd.notna(r.get(c)) else 0.0 for c in desc_cols]
                    X = np.hstack([fp, desc]).reshape(1, -1).astype(np.float64)
                    p = clf.predict_proba(scaler.transform(X))[0]
                    pred = float(p[pos])
                    if hasattr(clf, "estimators_") and clf.estimators_ is not None:
                        tree_ps = [t.predict_proba(scaler.transform(X))[0][pos]
                                   for t in clf.estimators_]
                        uncertainty = float(np.std(tree_ps))
                    max_sim_active = _max_tanimoto(fp, active_fps)
            murcko = str(r.get("murcko_scaffold", ""))
            score_rows.append({
                "panel_id": r["panel_id"],
                "predicted_a9a10_activity": pred,
                "tree_uncertainty": uncertainty,
                "max_tanimoto_to_a9a10_actives": max_sim_active,
                "novelty": (1.0 - max_sim_active) if pd.notna(max_sim_active) else np.nan,
                "scaffold_in_a9a10_training": int(murcko in train_scaffolds),
                "scaffold_in_candidate_families": int(murcko in fam_scaffolds),
            })
        scores = pd.DataFrame(score_rows)
        scores["scaffold_novelty"] = (1 - scores["scaffold_in_a9a10_training"]).astype(int)

        # ---- scaffold scan table ----
        scan_rows = []
        if self.families is not None and len(self.families):
            for _, f in self.families.iterrows():
                m = f["chemotype_scaffold"]
                hits = self.panel[self.panel["murcko_scaffold"].astype(str) == str(m)]
                scan_rows.append({
                    "family_id": f["family_id"],
                    "a9a10_candidate_chemotype": m,
                    "other_subtype_panel_overlap_count": int(len(hits)),
                    "overlapping_panel_ids": "|".join(hits["panel_id"].tolist()),
                })
        scan = pd.DataFrame(scan_rows) if scan_rows else pd.DataFrame(
            columns=["family_id", "a9a10_candidate_chemotype",
                     "other_subtype_panel_overlap_count", "overlapping_panel_ids"])

        # overlap of OTHER panel scaffolds with the a9a10 TRAINING scaffolds
        train_overlap = self.panel[
            self.panel["murcko_scaffold"].astype(str).isin(train_scaffolds)]
        summary = {
            "n_panel": int(len(self.panel)),
            "n_parseable": int(self.panel["is_parseable"].sum()),
            "n_a9a10_training_scaffold_hits": int(len(train_overlap)),
            "a9a10_training_scaffold_hit_ids": "|".join(train_overlap["panel_id"].tolist()),
            "n_candidate_family_overlaps": int(
                scan["other_subtype_panel_overlap_count"].sum()),
            "pred_activity_mean": float(scores["predicted_a9a10_activity"].mean()),
            "pred_activity_std": float(scores["predicted_a9a10_activity"].std()),
            "pred_activity_ge_0_5_frac": float(
                (scores["predicted_a9a10_activity"] >= 0.5).mean()),
        }
        return {"scores": scores, "scan": scan, "summary": summary}


def run_phase_selectivity(cfg_path: str) -> Dict[str, Any]:
    import sys
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("selectivity", str(cfg.resolve("logs")))
    panel_mod = OtherModulatorPanel(cfg, logger)

    sources = _discover_panel_sources(cfg)
    if not sources:
        logger.warn("No panel source CSVs found; selectivity panel empty")
        return {"panel": pd.DataFrame(), "scores": pd.DataFrame(),
                "scan": pd.DataFrame(), "physchem": pd.DataFrame(),
                "summary": {}, "isolation": {}}
    panel = panel_mod.curate(sources)

    train_path = cfg.resolve("data/processed/data_analysis_ready.csv")
    train = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
    fam_path = cfg.resolve("loop/library/chemotype_families.csv")
    families = pd.read_csv(fam_path) if fam_path.exists() else pd.DataFrame()
    artifact = str(cfg.resolve("models/classical/lbm_rf_fp_desc.joblib"))

    if not len(panel):
        logger.warn("Panel empty after merging sources")
        return {"panel": panel, "scores": pd.DataFrame(), "scan": pd.DataFrame(),
                "physchem": pd.DataFrame(), "summary": {}, "isolation": {}}

    bench = SelectivityBenchmark(cfg, logger, artifact, panel,
                                 train=train, families=families)
    res = bench.run()

    # ---- ISOLATION GUARD: exact-SMILES overlap vs alpha9alpha10 training ----
    isolation = {"contract": "panel stored under selectivity/ only; never written "
                            "to data/processed; source alpha9alpha10 training CSV "
                            "untouched"}
    if train is not None and len(train):
        panel_can = set(panel.dropna(subset=["canonical_SMILES"])["canonical_SMILES"])
        train_can = set(train.dropna(subset=["canonical_SMILES"])["canonical_SMILES"])
        overlap = sorted(panel_can & train_can)
        overlap_ids = panel[panel["canonical_SMILES"].isin(overlap)]["panel_id"].tolist()
        isolation["n_exact_smiles_overlap_with_training"] = int(len(overlap))
        isolation["overlap_panel_ids"] = "|".join(overlap_ids)
        if overlap:
            logger.warn(f"ISOLATION GUARD: {len(overlap)} panel SMILES occur in the "
                        f"a9a10 training set: {overlap_ids}")
        else:
            logger.info("Isolation guard OK: 0 exact-SMILES overlap with a9a10 training")
    else:
        isolation["n_exact_smiles_overlap_with_training"] = -1

    # ---- save outputs ----
    out = panel.drop(columns=["mol"], errors="ignore")
    save_df(out, str(cfg.resolve("selectivity/selectivity_panel.csv")))
    save_df(res["scores"], str(cfg.resolve("selectivity/selectivity_model_scores.csv")))
    save_df(res["scan"], str(cfg.resolve("selectivity/selectivity_scaffold_scan.csv")))

    # physchem coverage summary vs alpha9alpha10 actives + inactives
    rows = []
    if train is not None and len(train):
        for grp_name, sub in [("a9a10_actives", train[train["is_active"] == 1]),
                              ("a9a10_inactives", train[train["is_active"] == 0])]:
            if not len(sub):
                continue
            for dc in ["MW", "LogP", "TPSA", "HBA", "RotBonds", "RingCount",
                       "AromaticRings", "FractionCSP3"]:
                lo = float(sub[dc].min()); hi = float(sub[dc].max())
                in_range = float(panel[dc].between(lo, hi).mean())
                rows.append({"group": grp_name, "descriptor": dc,
                             "train_min": lo, "train_max": hi,
                             "frac_panel_in_train_range": in_range})
    phys = pd.DataFrame(rows)
    save_df(phys, str(cfg.resolve("selectivity/selectivity_physchem_summary.csv")))

    save_manifest(
        make_provenance("selectivity", cfg, {
            "raw_other_modulators_hash": panel_mod.raw_hash,
            "source_hashes": panel_mod.source_hashes,
            "source_counts": panel_mod.source_count,
            "n_deduped_by_canonical_smiles": panel_mod.n_deduped,
            "panel_data_hash": panel_mod.panel_hash,
            "n_panel": int(len(panel)),
            "n_parseable": int(panel["is_parseable"].sum()),
            "isolation": isolation,
            "summary": res["summary"],
        }),
        str(cfg.resolve("selectivity/selectivity_manifest.json")),
    )
    logger.info(f"Panel summary: {json.dumps(res['summary'], default=str)}")
    logger.info(f"Isolation guard: {json.dumps(isolation, default=str)}")

    # write the human-readable report (same as main())
    report = write_report(cfg, {"panel": panel, "scores": res["scores"],
                                "scan": res["scan"], "physchem": phys,
                                "summary": res["summary"]})
    out_md = cfg.resolve("selectivity/selectivity_report.md")
    os.makedirs(os.path.dirname(str(out_md)), exist_ok=True)
    with open(str(out_md), "w", encoding="utf-8") as f:
        f.write(report)

    return {"panel": panel, "scores": res["scores"], "scan": res["scan"],
            "physchem": phys, "summary": res["summary"], "isolation": isolation}


def write_report(cfg: ProjectConfig, res: Dict[str, Any]) -> str:
    panel = res["panel"]
    scores = res["scores"]
    scan = res["scan"]
    phys = res["physchem"]
    s = res["summary"]
    isolation = res.get("isolation", {})

    m = pd.merge(panel, scores, on="panel_id", how="left")
    m = m.sort_values("predicted_a9a10_activity", ascending=False)

    def fmt(x):
        return "%.3f" % x if pd.notna(x) else "NA"

    lines = []
    lines.append("# Non-alpha9alpha10 nAChR modulator selectivity panel")
    lines.append("")
    lines.append(f"_Generated automatically by `project/src/other_modulators.py` "
                 f"({cfg.project.get('name','')})._")
    lines.append("")
    lines.append("## 1. Motivation / data-integrity note")
    lines.append("")
    lines.append("The compounds in `<repo>/otherModulators.csv` are modulators of **other** "
                 "nAChR subtypes (alpha7 and alpha4beta2 PAMs, the CMPI/isoxazole-indole "
                 "series, and nAChR-active flavonoids/natural products). They are **not** "
                 "alpha9alpha10 measurements. To avoid corrupting the alpha9alpha10 "
                 "`is_active` training target, they are kept as a separate, documented "
                 "**selectivity / decoys panel** rather than merged into the training set. "
                 "Uses:")
    lines.append("- **Decoys / selectivity modeling:** negative or off-target signal for "
                 "future selectivity-aware models and as a panel designed a9alpha10 hits "
                 "should discriminate against.")
    lines.append("- **Chemotype scan:** Bemis-Murcko scaffold overlap vs the ascorbate "
                 "a9alpha10 library and candidate chemotype families.")
    lines.append("- **Benchmark:** frozen a9alpha10 model predictions on the panel define "
                 "the current selectivity baseline.")
    lines.append("")
    lines.append("## 2. Panel summary")
    lines.append("")
    lines.append(f"- Total panel: **{int(s['n_panel'])}**; parseable: **{int(s['n_parseable'])}**")
    lines.append(f"- Share with a9alpha10-training scaffolds: **{int(s['n_a9a10_training_scaffold_hits'])}** "
                 f"({s['a9a10_training_scaffold_hit_ids'] or 'none'})")
    lines.append(f"- Overlaps with candidate chemotype families: **{int(s['n_candidate_family_overlaps'])}**")
    if isolation.get("n_exact_smiles_overlap_with_training") is not None:
        n_overlap = isolation["n_exact_smiles_overlap_with_training"]
        lines.append(f"- Isolation guard (exact-SMILES overlap with alpha9alpha10 training): "
                     f"**{n_overlap}** "
                     f"({isolation.get('overlap_panel_ids') or 'none'}); "
                     f"{isolation.get('contract', '')}")
    lines.append("")
    lines.append("## 3. Frozen a9alpha10 model on the panel (selectivity baseline)")
    lines.append("")
    lines.append(f"- Mean predicted a9alpha10 PAM activity: **{fmt(s['pred_activity_mean'])}** "
                 f"± {fmt(s['pred_activity_std'])}")
    lines.append(f"- Fraction predicted active (>=0.5): **{fmt(s['pred_activity_ge_0_5_frac'])}**")
    lines.append("")
    lines.append("> Interpretation: the frozen a9alpha10 model was trained only on the "
                 "ascorbate chemistry. A low predicted-activity / low blacklisted fraction "
                 "for this panel is the desired selectivity behaviour; a high fraction means "
                 "designed hits could cross-react. The 7 a9alpha10 actives (ascorbate core) "
                 "should separate clearly from this panel.")
    lines.append("")
    lines.append("### Top-predicted panel compounds (potential selectivity watch-list)")
    lines.append("")
    lines.append("| panel_id | predicted_a9a10 | novelty | scaffold_in_a9a10 | source_label | subtype_note |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in m.head(15).iterrows():
        lines.append(f"| {r['panel_id']} | {fmt(r['predicted_a9a10_activity'])} | "
                     f"{fmt(r['novelty'])} | {int(r['scaffold_in_a9a10_training'])} | "
                     f"{r['source_label']} | {r.get('subtype_note','')} |")
    lines.append("")
    lines.append("## 4. Chemotype (Bemis-Murcko) scan vs ascorbate candidates")
    lines.append("")
    if len(scan):
        lines.append("Overlap of other-subtype panel scaffolds with the a9alpha10 candidate "
                     "chemotype families (a non-empty overlap flags a scaffold the alpha9alpha10 "
                     "library shares with other-subtype modulators):")
        lines.append("")
        lines.append("| family_id | a9alpha10 candidate chemotype | panel overlap count | overlapping panel IDs |")
        lines.append("|---|---|---|---|")
        for _, r in scan.iterrows():
            if int(r["other_subtype_panel_overlap_count"]) > 0:
                lines.append(f"| {r['family_id']} | `{r['a9a10_candidate_chemotype']}` | "
                             f"{int(r['other_subtype_panel_overlap_count'])} | "
                             f"{r['overlapping_panel_ids']} |")
    else:
        lines.append("No overlap detected between the other-subtype panel scaffolds and the "
                     "a9alpha10 candidate chemotype families.")
    lines.append("")
    lines.append("## 5. Physicochemical coverage of the panel relative to a9alpha10 library")
    lines.append("")
    lines.append("Fraction of the other-subtype panel that falls inside the a9alpha10 "
                 "training descriptor range (per descriptor, per a9alpha10 activity group).")
    lines.append("")
    lines.append("| group | descriptor | train_min | train_max | frac_panel_in_train_range |")
    lines.append("|---|---|---|---|---|")
    for _, r in phys.iterrows():
        lines.append(f"| {r['group']} | {r['descriptor']} | {fmt(r['train_min'])} | "
                     f"{fmt(r['train_max'])} | {fmt(r['frac_panel_in_train_range'])} |")
    lines.append("")
    lines.append("## 6. Isolation contract")
    lines.append("")
    lines.append("- This panel is stored under `<project>/selectivity/` only. It is **never** "
                 "written into `data/processed/` and the alpha9alpha10 training CSV "
                 "(`modulator-dataset-a9a10.csv`) is never modified.")
    lines.append("- An isolation guard compares canonical SMILES between the panel and the "
                 "alpha9alpha10 training set on every run and reports any exact overlap in "
                 "`selectivity/selectivity_manifest.json` (`isolation` block).")
    lines.append("")
    lines.append("## 7. Expanding the panel later")
    lines.append("")
    lines.append("The panel is deliberately small and can be extended without touching code:")
    lines.append("- **Drop-in files:** add any number of CSV files to "
                 "`<project>/data/selectivity/raw/` with the same columns as `otherModulators.csv` "
                 "(`File Name, cleanedMol, classLabel`). Re-run the phase and they are merged "
                 "automatically (each file is versioned by content hash in the manifest).")
    lines.append("- **Config override:** optionally list extra files under "
                 "`config.yaml -> data -> selectivity_panel -> sources`.")
    lines.append("- **Deduplication:** rows with identical canonical SMILES are merged "
                 "(first-seen source wins); per-source counts and the number deduped are recorded "
                 "in the manifest.")
    lines.append("- New chemotypes feed the same Bemis-Murcko scan and frozen-model selectivity "
                 "benchmark; keep this section of the report as the living baseline.")
    lines.append("")
    lines.append("## 8. Follow-up / honest caveats")
    lines.append("")
    lines.append("- `source_label` is carried from the source file verbatim and is **not** an "
                 "alpha9alpha10 measurement; treat only as a within-source binary class.")
    lines.append("- Subtype notes are literature-informed annotations for interpretability; "
                 "verify against primary references before publication.")
    lines.append("- This panel has NOT been merged into the alpha9alpha10 training set by design.")
    lines.append("")
    return "\n".join(lines)


def main():  # pragma: no cover
    import sys
    default_cfg = str(Path("/Users/nb/Documents/ligand-based-modeling/project/configs/config.yaml"))
    if len(sys.argv) > 1:
        default_cfg = sys.argv[1]
    from pathlib import Path as _P
    cfg = ProjectConfig(default_cfg)
    res = run_phase_selectivity(default_cfg)
    report = write_report(cfg, res)
    out = cfg.resolve("selectivity/selectivity_report.md")
    os.makedirs(os.path.dirname(str(out)), exist_ok=True)
    with open(str(out), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote selectivity_report.md -> {out}")
    print(f"Wrote {len(res['panel'])} panel rows, {len(res['scores'])} score rows")


if __name__ == "__main__":
    main()
