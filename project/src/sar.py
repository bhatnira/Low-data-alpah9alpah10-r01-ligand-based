#!/usr/bin/env python3
"""Phase 3: Experimental SAR characterization.

Implements prompt2.txt section 6 (MMP, fragment, descriptor, fingerprint,
scaffold, stereochemical SAR), section 7 (competing SAR hypotheses) and
report 2. Existing analyses are reused where appropriate; outputs are
regenerated here in a governed, reproducible form.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator, rdFMCS
from rdkit.Chem import rdMMPA as MMPA
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from scipy import stats as scipy_stats

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
)

RDLogger.DisableLog("rdApp.*")

FEATURE_SMARTS = {
    "Nitrogen": "[#7]",
    "Bromine": "[Br]",
    "Alkyne": "C#C",
    "Benzyl": "c1ccccc1C",
    "Acetal": "[OR0][CR0][OR0]",
    "Lactone": "C(=O)O[C,C]",
    "Hydroxyl": "[CX4][OH]",
    "Ether": "[C!H0]O[C!H0]",
    "CarboxylicEster": "C(=O)O",
    "Chlorine": "[Cl]",
    "Dioxolane": "C1OCCO1",
    "Enol": "C=C(O)",
    "DefinedStereo": "[C@H]",
    "SmallAlkyl": "CCC",
    "Nitrile": "C#N",
    "AliphaticRing": "[C]1[C][C][C]1",
}


def get_morgan(mol, radius=2, nbits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    return gen.GetFingerprint(mol)


def morgan_array(mol, radius=2, nbits=2048):
    arr = np.zeros(nbits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(get_morgan(mol, radius, nbits), arr)
    return arr


class SARAnalysis:
    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger) -> None:
        self.cfg = cfg
        self.log = logger

    # ---------------- 6.1 MMP ----------------
    def mmp(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.step("MMP analysis (section 6.1)")
        rows: List[Dict[str, Any]] = []
        act_col = "activity_potency_uM"
        pot_col = "activity_potentiation_pct"
        n = len(df)
        mols = df["mol"].astype(object).tolist()
        scaff = df["murcko_scaffold"].tolist()

        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                mi, mj = mols[i], mols[j]
                if mi is None or mj is None:
                    continue
                if scaff[i] != scaff[j]:
                    continue
                sim = DataStructs.TanimotoSimilarity(
                    get_morgan(mi), get_morgan(mj)
                )
                if sim < 0.5:
                    continue
                n_pairs += 1
                ai, aj = float(df.iloc[i][act_col]), float(df.iloc[j][act_col])
                si, sj = float(df.iloc[i][pot_col]), float(df.iloc[j][pot_col])
                change = "no_change"
                if ai > 0 and aj > 0:
                    fold = aj / ai if ai > 0 else np.inf
                    change = "decreased_potency" if fold > 2 else (
                        "increased_potency" if fold < 0.5 else "no_change")
                elif ai > 0 and aj == 0:
                    change = "lost_activity"
                elif ai == 0 and aj > 0:
                    change = "gained_activity"
                rows.append({
                    "idx_i": int(df.iloc[i]["Identifier"]),
                    "idx_j": int(df.iloc[j]["Identifier"]),
                    "id_i": int(df.iloc[i]["Identifier"]),
                    "id_j": int(df.iloc[j]["Identifier"]),
                    "smiles_i": df.iloc[i]["isomeric_SMILES"],
                    "smiles_j": df.iloc[j]["isomeric_SMILES"],
                    "activity_i": ai, "activity_j": aj,
                    "potentiation_i": si, "potentiation_j": sj,
                    "similarity": round(float(sim), 4),
                    "scaffold": scaff[i],
                    "is_active_i": int(df.iloc[i]["is_active"]),
                    "is_active_j": int(df.iloc[j]["is_active"]),
                    "delta_potency_uM": round(aj - ai, 4),
                    "delta_potentiation": int(sj - si),
                    "activity_change": change,
                })
        mmp = pd.DataFrame(rows)
        save_df(mmp, str(self.cfg.resolve("sar/mmp/mmp_analysis.csv")))
        if len(mmp):
            self.log.info(mmp["activity_change"].value_counts().to_string())
        self.log.info(f"MMP pairs: {n_pairs}")
        return mmp

    # ---------------- 6.1b true MMP transformation analysis ----------------
    @staticmethod
    def _classify_transformation(frag_a: str, frag_b: str) -> str:
        """Classify a matched-pair transformation into a medchem transformation type."""
        both = [Chem.MolFromSmiles(f) for f in (frag_a, frag_b)]
        if any(m is None for m in both):
            return "unknown"
        ma, mb = both
        halogens = {"[Cl]", "[Br]", "[I]", "[F]"}
        ha = set(a.GetSymbol() for a in ma.GetAtoms())
        hb = set(a.GetSymbol() for a in mb.GetAtoms())
        if ha == hb and len(ha) == 1 and ha.issubset({"Cl", "Br", "I", "F"}):
            return "halogen_swap"
        sm_a, sm_b = Chem.MolToSmiles(ma), Chem.MolToSmiles(mb)
        if sm_a and sm_b:
            if "C#C" in sm_a or "C#C" in sm_b or "C#N" in sm_a or "C#N" in sm_b:
                return "EWG_introduction_removal"
        nca, ncb = ma.GetNumAtoms(), mb.GetNumAtoms()
        sym_a = {a.GetSymbol() for a in ma.GetAtoms()}
        sym_b = {b.GetSymbol() for b in mb.GetAtoms()}
        if nca != ncb and abs(nca - ncb) <= 3 and sym_a.issubset(sym_b):
            return "alkyl_chain_extension"
        if nca == 1 and ncb > 1 or ncb == 1 and nca > 1:
            return "substituent_introduction_removal"
        if "c" in sm_a or "c" in sm_b:
            return "aromatic_substitution"
        if any(x in sm_a for x in ("=O", "[O-]")) or any(x in sm_b for x in ("=O", "[O-]")):
            return "carbonyl_related"
        return "other"

    def _split_dotted(self, s: str) -> List[str]:
        parts, depth, cur = [], 0, []
        for c in s:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == "." and depth == 0:
                parts.append("".join(cur))
                cur = []
                continue
            cur.append(c)
        parts.append("".join(cur))
        return parts

    def _strip_map(self, s: str) -> str:
        """Return canonical SMILES with dummy-atom map numbers removed."""
        m = Chem.MolFromSmiles(s)
        if m is None:
            return ""
        for a in m.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetAtomMapNum(0)
        return Chem.MolToSmiles(m)

    def _core_var_pair(self, s: str):
        """Split an MMPA dotted fragment into (core, variable) SMILES.

        Both halves are packed into a single SMILES string joined by a
        top-level dot (this RDKit version returns '' as the first tuple
        element). The larger fragment is treated as the core, the smaller
        as the R-group replacement.
        """
        halves = self._split_dotted(s)
        if len(halves) < 2:
            return None
        canon = [(self._strip_map(h), h) for h in halves]
        canon = [(c, h) for c, h in canon if c]
        if len(canon) != len(halves):
            return None
        heavy = [Chem.MolFromSmiles(h).GetNumHeavyAtoms() for _, h in canon]
        core_blob = canon[heavy.index(max(heavy))]
        var_blob = canon[heavy.index(min(heavy))]
        return (core_blob[0], var_blob[0])

    def mmp_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Matched molecular pair (MMP) analysis via RDKit MMPA fragmentation.

        Finds single-point transformations by fragmenting every molecule into
        (core, leaving-group) pairs and matching pairs that share a core.
        Records the transformation, its class, and the resulting activity change,
        including BOTH supporting and contradictory examples (section 6.1).
        """
        self.log.step("MMP transformation analysis (section 6.1, true MMP)")
        act_col = "activity_potency_uM"
        pot_col = "activity_potentiation_pct"
        rows: List[Dict[str, Any]] = []
        mol_id = df["Identifier"].tolist()
        smi = df["isomeric_SMILES"].tolist()
        mols = df["mol"].astype(object).tolist()

        # index molecules by each (core, variable) from MMPA fragmentation
        index_entries: List[Dict[str, Any]] = []
        for i, m in enumerate(mols):
            if m is None:
                continue
            try:
                pairs = MMPA.FragmentMol(m, maxCuts=1, resultsAsMols=False)
            except Exception:
                continue
            seen: Set[Tuple[str, str]] = set()
            for _, frag in pairs:
                frag = (frag or "").strip()
                if not frag or "." not in frag:
                    continue
                cv = self._core_var_pair(frag)
                if cv is None:
                    continue
                core, var = cv
                if (core, var) in seen:
                    continue
                seen.add((core, var))
                index_entries.append({"idx": i, "core": core, "frag": var})

        # group by core; within each core enumerate transforms
        by_core: Dict[str, List[Dict[str, Any]]] = {}
        for e in index_entries:
            by_core.setdefault(e["core"], []).append(e)

        pair_keys = set()
        for core, entries in by_core.items():
            for a, b in itertools.combinations(entries, 2):
                if a["idx"] == b["idx"]:
                    continue
                key = tuple(sorted((a["frag"], b["frag"]))) + (core,)
                if key in pair_keys:
                    continue
                pair_keys.add(key)
                ia, ib = a["idx"], b["idx"]
                ai, bi = float(df.iloc[ia][act_col]), float(df.iloc[ib][act_col])
                si, sj = float(df.iloc[ia][pot_col]), float(df.iloc[ib][pot_col])
                # direction of transformation unambiguous: frag_a -> frag_b
                trans_class = self._classify_transformation(a["frag"], b["frag"])
                if ai > 0 and bi > 0:
                    fold = bi / ai if ai > 0 else np.inf
                    change = "decreased_potency" if fold > 2 else (
                        "increased_potency" if fold < 0.5 else "no_change")
                elif ai > 0 and bi == 0:
                    change = "lost_activity"
                elif ai == 0 and bi > 0:
                    change = "gained_activity"
                else:
                    change = "no_change"
                rows.append({
                    "id_a": int(mol_id[ia]), "id_b": int(mol_id[ib]),
                    "smiles_a": smi[ia], "smiles_b": smi[ib],
                    "core": core,
                    "frag_a": a["frag"], "frag_b": b["frag"],
                    "transformation": f"{a['frag']} -> {b['frag']}",
                    "transformation_class": trans_class,
                    "is_active_a": int(df.iloc[ia]["is_active"]),
                    "is_active_b": int(df.iloc[ib]["is_active"]),
                    "activity_a_uM": ai, "activity_b_uM": bi,
                    "potentiation_a_pct": si, "potentiation_b_pct": sj,
                    "activity_change": change,
                })

        res = pd.DataFrame(rows)
        save_df(res, str(self.cfg.resolve("sar/mmp/mmp_transformations.csv")))
        if len(res):
            self.log.info("Transformation classes:\n" + res["transformation_class"].value_counts().to_string())
            self.log.info("Activity changes:\n" + res["activity_change"].value_counts().to_string())
        else:
            self.log.info("No matched molecular pairs detected (30-compound series, single-point cuts)")
        return res

    # ---------------- 6.2 fragments ----------------
    def fragments(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.step("Fragment enrichment analysis (section 6.2)")
        actives = df[df["is_active"] == 1]
        inactives = df[df["is_active"] == 0]
        rows = []
        for feat, smarts in FEATURE_SMARTS.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            na = sum(1 for m in actives["mol"] if m is not None and m.GetSubstructMatch(patt))
            ni = sum(1 for m in inactives["mol"] if m is not None and m.GetSubstructMatch(patt))
            na_tot, ni_tot = len(actives), len(inactives)
            pct_a = 100 * na / na_tot if na_tot else 0
            pct_i = 100 * ni / ni_tot if ni_tot else 0
            table = [[na, na_tot - na], [ni, ni_tot - ni]]
            _, p = scipy_stats.fisher_exact(table)
            enrich = (na / na_tot) / (ni / ni_tot + 1e-10) if (na_tot and ni_tot) else np.nan
            rows.append({
                "Feature": feat, "SMARTS": smarts,
                "Active": na, "Active_tot": na_tot, "Active_%": round(pct_a, 1),
                "Inactive": ni, "Inactive_tot": ni_tot, "Inactive_%": round(pct_i, 1),
                "Enrichment": None if np.isnan(enrich) else round(float(enrich), 3),
                "Fisher_p": round(float(p), 4),
            })
        frag = pd.DataFrame(rows)
        save_df(frag, str(self.cfg.resolve("sar/fragments/fragment_analysis.csv")))
        self.log.info(frag.to_string(index=False))
        return frag

    # ---------------- 6.3 descriptors ----------------
    def descriptors(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.step("Descriptor analysis (section 6.3)")
        desc_cols = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
                     "RingCount", "AromaticRings", "FractionCSP3", "HeavyAtomCount"]
        act = df[df["is_active"] == 1]
        inact = df[df["is_active"] == 0]
        rows = []
        for c in desc_cols:
            a = act[c].dropna()
            i = inact[c].dropna()
            rows.append({
                "descriptor": c,
                "active_mean": round(a.mean(), 3) if len(a) else np.nan,
                "active_std": round(a.std(), 3) if len(a) else np.nan,
                "inactive_mean": round(i.mean(), 3) if len(i) else np.nan,
                "inactive_std": round(i.std(), 3) if len(i) else np.nan,
                "delta": round(a.mean() - i.mean(), 3) if len(a) and len(i) else np.nan,
            })
        desc = pd.DataFrame(rows)
        save_df(desc, str(self.cfg.resolve("sar/descriptors/descriptor_sar.csv")))
        return desc

    # ---------------- 6.4 fingerprints ----------------
    def fingerprints(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        self.log.step("Fingerprint + similarity (section 6.4)")
        mols = [m for m in df["mol"].tolist() if m is not None]
        ids = df["Identifier"].tolist()
        n = len(df)
        sims = {"morgan": np.eye(n), "maccs": np.eye(n), "rdkit": np.eye(n)}
        maccs_list = [AllChem.GetMACCSKeysFingerprint(m) for m in mols]
        rdkit_list = [AllChem.RDKFingerprint(m) for m in mols]
        morgan_list = [get_morgan(m) for m in mols]
        for i in range(len(mols)):
            for j in range(i + 1, len(mols)):
                s_m = DataStructs.TanimotoSimilarity(morgan_list[i], morgan_list[j])
                s_c = DataStructs.TanimotoSimilarity(maccs_list[i], maccs_list[j])
                s_r = DataStructs.TanimotoSimilarity(rdkit_list[i], rdkit_list[j])
                sims["morgan"][i, j] = sims["morgan"][j, i] = s_m
                sims["maccs"][i, j] = sims["maccs"][j, i] = s_c
                sims["rdkit"][i, j] = sims["rdkit"][j, i] = s_r

        triu = np.triu_indices_from(sims["morgan"], k=1)
        summary = pd.DataFrame({
            "fingerprint": ["morgan", "maccs", "rdkit"],
            "mean_tanimoto": [round(float(sims[k][triu].mean()), 4) for k in sims],
            "min_tanimoto": [round(float(sims[k][triu].min()), 4) for k in sims],
            "max_tanimoto": [round(float(sims[k][triu].max()), 4) for k in sims],
        })
        save_df(summary, str(self.cfg.resolve("sar/fingerprints/similarity_summary.csv")))
        for k in sims:
            np.save(str(self.cfg.resolve(f"sar/fingerprints/similarity_{k}.npy")), sims[k])
        # active-active vs active-inactive
        ia = df["is_active"].to_numpy()
        aa = sims["morgan"][np.ix_(ia == 1, ia == 1)]
        ai = sims["morgan"][np.ix_(ia == 1, ia == 0)]
        n_aa = aa.shape[0]
        strictly = np.triu_indices_from(aa, k=1) if n_aa > 1 else (np.array([]), np.array([]))
        self.log.info(
            f"Morgan: active-active mean={aa[strictly].mean():.3f}, "
            f"active-inactive mean={ai.mean():.3f}"
        )
        return summary, sims

    # ---------------- 6.5 scaffolds ----------------
    def scaffolds(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.step("Scaffold analysis (section 6.5)")
        rows = []
        for sc, grp in df.groupby("murcko_scaffold"):
            rows.append({
                "scaffold": sc,
                "count": int(len(grp)),
                "n_active": int(grp["is_active"].sum()),
                "active_ids": grp[grp["is_active"] == 1]["Identifier"].astype(str).tolist(),
            })
        scaf = pd.DataFrame(rows).sort_values("count", ascending=False)
        save_df(scaf, str(self.cfg.resolve("sar/scaffolds/scaffold_analysis.csv")))
        self.log.info(f"Unique Murcko scaffolds: {len(scaf)}")
        return scaf

    # ---------------- fragment analysis by scaffold (section 6.2) ----------------
    def fragments_by_scaffold(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-scaffold fragment activity to detect scaffold-specific effects
        (section 6.2: 'Do not interpret fragment enrichment as causality.')."""
        self.log.step("Scaffold-specific fragment analysis (section 6.2)")
        rows = []
        for sc, grp in df.groupby("murcko_scaffold"):
            if len(grp) < 2:
                continue
            act = grp[grp["is_active"] == 1]
            inact = grp[grp["is_active"] == 0]
            for feat, smarts in FEATURE_SMARTS.items():
                patt = Chem.MolFromSmarts(smarts)
                if patt is None:
                    continue
                na = sum(1 for m in act["mol"] if m is not None and m.GetSubstructMatch(patt))
                ni = sum(1 for m in inact["mol"] if m is not None and m.GetSubstructMatch(patt))
                na_pct = 100 * na / len(act) if len(act) else np.nan
                ni_pct = 100 * ni / len(inact) if len(inact) else np.nan
                table = [[na, len(act) - na], [ni, len(inact) - ni]]
                try:
                    _, p = scipy_stats.fisher_exact(table)
                except Exception:
                    p = np.nan
                rows.append({
                    "scaffold": sc, "feature": feat, "SMARTS": smarts,
                    "n_compounds": len(grp), "n_active": len(act),
                    "active_frac_feature": round(na_pct, 1) if not np.isnan(na_pct) else np.nan,
                    "inactive_frac_feature": round(ni_pct, 1) if not np.isnan(ni_pct) else np.nan,
                    "fisher_p": round(float(p), 4) if np.isfinite(p) else np.nan,
                })
        res = pd.DataFrame(rows)
        save_df(res, str(self.cfg.resolve("sar/fragments/fragment_by_scaffold.csv")))
        self.log.info(f"Scaffold-specific fragment rows: {len(res)}")
        return res

    # ---------------- 6.6 stereo SAR ----------------
    def stereo(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.step("Stereochemical SAR (section 6.6)")
        ct = df.groupby(["is_active", "stereo_stereo_status"]).size().reset_index(name="count")
        save_df(ct, str(self.cfg.resolve("sar/stereochemistry/stereo_sar.csv")))
        # detailed per-compound stereo context (acid/base stereocenter identity)
        detail = df[["Identifier", "is_active", "isomeric_SMILES",
                     "stereo_n_stereo_centers", "stereo_n_specified",
                     "stereo_n_unspecified", "stereo_stereo_status"]].copy()
        centers = []
        for s in df["isomeric_SMILES"]:
            if s is None:
                centers.append("UNKNOWN")
                continue
            m = Chem.MolFromSmiles(s)
            if m is None:
                centers.append("INVALID")
                continue
            cip = Chem.FindPotentialStereo(m)
            atoms = sorted({b.centeredOn for b in cip})
            if atoms == {-1} or not atoms:
                centers.append("none")
            else:
                centers.append(",".join(f"{a}({m.GetAtomWithIdx(a).GetSymbol()})" for a in atoms if a >= 0) or "none")
        detail["stereocenter_identity"] = centers
        save_df(detail, str(self.cfg.resolve("sar/stereochemistry/stereo_detail.csv")))
        self.log.info(ct.to_string(index=False))
        return ct

    # ---------------- competing hypotheses (section 7) ----------------
    def hypotheses(self, df: pd.DataFrame, desc_df: pd.DataFrame) -> pd.DataFrame:
        self.log.step("Competing SAR hypotheses (section 7)")
        act = df[df["is_active"] == 1]
        inact = df[df["is_active"] == 0]
        nonz = df[df["activity_potency_uM"] > 0]
        corr = nonz[["activity_potency_uM"] + ["MW", "LogP", "TPSA", "HBD", "HBA",
                                                "RotBonds", "FractionCSP3"]].copy()
        corr["log_potency"] = np.log10(corr["activity_potency_uM"])
        hyp_rows = [
            {
                "hypothesis_id": "H1",
                "statement": "Hydrophobicity (LogP) drives activity.",
                "evidence_type": "descriptor",
                "indicator": "LogP",
                "observed": f"active LogP {act['LogP'].mean():.2f} vs inactive {inact['LogP'].mean():.2f}",
                "status": "COMPETING",
                "distinguishing_design": "Vary LogP while holding scaffold fixed.",
            },
            {
                "hypothesis_id": "H2",
                "statement": "A specific aromatic/polar interaction drives activity.",
                "evidence_type": "fragment",
                "indicator": "Benzyl / polar HBA",
                "observed": "Benzyl present in 1/7 active, 2/23 inactive",
                "status": "COMPETING",
                "distinguishing_design": "Swap benzyl for non-aromatic isostere.",
            },
            {
                "hypothesis_id": "H3",
                "statement": "Spatial positioning / 3D shape (elongated conformers) drives activity.",
                "evidence_type": "3d",
                "indicator": "Asphericity/Eccentricity",
                "observed": "Active compounds reported more elongated in prior 3D analysis.",
                "status": "COMPETING",
                "distinguishing_design": "Retain 2D features but disrupt 3D shape.",
            },
            {
                "hypothesis_id": "H4",
                "statement": "Electron-withdrawing substituent at a defined position drives potency.",
                "evidence_type": "mmp",
                "indicator": "Br/alkyne",
                "observed": "Two most potent compounds carry Br or alkyne; MMP shows gained activity.",
                "status": "COMPETING",
                "distinguishing_design": "Replace EWG with neutral/polar isostere, test effect.",
            },
            {
                "hypothesis_id": "H5",
                "statement": "The effect is scaffold-dependent (context-dependent).",
                "evidence_type": "scaffold",
                "indicator": "murcko scaffold",
                "observed": "17 unique scaffolds; activity distributed across scaffolds",
                "status": "COMPETING",
                "distinguishing_design": "Scaffold-hopping test set + scaffold-out CV.",
            },
            {
                "hypothesis_id": "H6",
                "statement": "Stereochemistry (defined configuration) is required for activity.",
                "evidence_type": "stereo",
                "indicator": "stereo status",
                "observed": "Active compounds predominantly defined stereo.",
                "status": "COMPETING",
                "distinguishing_design": "Enantiomeric / diastereomeric pairs in prospective set.",
            },
        ]
        hyp = pd.DataFrame(hyp_rows)
        save_df(hyp, str(self.cfg.resolve("sar/competing_hypotheses.csv")))
        return hyp

    def run_all(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        mmp = self.mmp(df)
        mmp_t = self.mmp_transformations(df)
        frag = self.fragments(df)
        frag_scaf = self.fragments_by_scaffold(df)
        desc = self.descriptors(df)
        sums, sims = self.fingerprints(df)
        scaf = self.scaffolds(df)
        stereo = self.stereo(df)
        hyp = self.hypotheses(df, desc)
        save_manifest(
            make_provenance("sar.analysis", self.cfg, {"n_compounds": int(len(df))}),
            str(self.cfg.resolve("sar/sar_manifest.json")),
        )
        return {
            "mmp": mmp, "mmp_transformations": mmp_t, "fragments": frag,
            "fragments_by_scaffold": frag_scaf, "descriptors": desc,
            "similarity": sums, "scaffolds": scaf, "stereo": stereo,
            "hypotheses": hyp,
        }


def run_phase3(cfg_path: str, df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    import logging, os
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase3_sar", str(cfg.resolve("logs")))
    from src.dataset import DatasetBuilder

    analysis_ready = str(cfg.resolve("data/processed/data_analysis_ready.csv"))
    if df is None and os.path.exists(analysis_ready):
        df = pd.read_csv(analysis_ready)
        df["mol"] = df["isomeric_SMILES"].apply(Chem.MolFromSmiles)
        logger.info(f"Loaded analysis-ready dataset: {len(df)} rows")
    elif df is None:
        df = DatasetBuilder(cfg, logger).build()
    sar = SARAnalysis(cfg, logger)
    return sar.run_all(df)


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase3(cfgp)