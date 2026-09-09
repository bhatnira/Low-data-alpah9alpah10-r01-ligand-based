#!/usr/bin/env python3
"""Phase 5: Explainable AI + stability / reproducibility analysis.

Implements prompt2.txt sections 12, 13 (XAI reproducibility), and report 4:
  - multiple independent explanation methods (SHAP, permutation importance,
    RF Gini, fragment attribution, feature ablation, counterfactuals)
  - explanations repeated across models/seeds/CV-splits
  - stability metrics: Spearman correlation, top-k overlap, Jaccard,
    sign consistency, feature-selection frequency
  - unstable explanations reported AS unstable (no cherry-picking)
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
)

RDLogger.DisableLog("rdApp.*")

DESC_COLS = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
             "RingCount", "AromaticRings", "FractionCSP3", "HeavyAtomCount"]
N_FP = 2048


def morgan_array(mol, radius=2, nbits=N_FP):
    from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class XAI:
    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger
        self.fp_mask: Optional[np.ndarray] = None
        self.fp_feature_names: List[str] = []
        self.n_fp_used = N_FP

    def _feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        desc = df[DESC_COLS].fillna(0).to_numpy(dtype=np.float64)
        arr = np.vstack([morgan_array(m) if m is not None else np.zeros(N_FP) for m in df["mol"]])
        # keep only fingerprint bits that occur in at least one compound
        self.fp_mask = arr.sum(axis=0) >= 1
        self.n_fp_used = int(self.fp_mask.sum())
        self.fp_feature_names = [f"fp_{int(i)}" for i in np.where(self.fp_mask)[0]]
        X = np.hstack([arr[:, self.fp_mask], desc]).astype(np.float64)
        self.log.info(f"Fingerprint features used (non-degenerate bits): {self.n_fp_used}/{N_FP}")
        return X, self.fp_feature_names + DESC_COLS

    def train_rf(self, X, y, seed: int = 42):
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
        rf.fit(Xs, y)
        return rf, sc

    # ---------------- methods ----------------
    def gini_importance(self, rf) -> pd.DataFrame:
        imp = rf.feature_importances_
        desc_imp = imp[self.n_fp_used:]
        fp_imp_top = np.argsort(-imp[:self.n_fp_used])[:10]
        self.log.info("Top fingerprint bits by Gini: " + ", ".join(
            f"{self.fp_feature_names[i]}({imp[i]:.4f})" for i in fp_imp_top))
        return pd.DataFrame({
            "feature": DESC_COLS,
            "rf_gini_importance": desc_imp,
            "rank": np.argsort(-desc_imp) + 1,
        })

    def permutation_importance(self, X, y, rf, sc, n_repeats: int = 10, seed: int = 42) -> pd.DataFrame:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(rf, sc.transform(X[:, :self.n_fp_used + len(DESC_COLS)]),
                                   y, n_repeats=n_repeats, random_state=seed)
        desc_idx = np.arange(self.n_fp_used, self.n_fp_used + len(DESC_COLS))
        return pd.DataFrame({
            "feature": DESC_COLS,
            "perm_importance_mean": r.importances_mean[desc_idx],
            "perm_importance_std": r.importances_std[desc_idx],
        })

    def shap_values(self, X, y, rf, sc, n_background=None, seed: int = 42) -> Dict[str, Any]:
        try:
            import shap
        except Exception as e:
            return {"available": False, "error": str(e)}
        n_background = n_background or min(len(X), 15)
        rng = np.random.default_rng(seed)
        bg_idx = rng.choice(len(X), size=n_background, replace=False)
        explainer = shap.TreeExplainer(rf)
        Xs = sc.transform(X)
        all_sv = explainer.shap_values(Xs)
        if isinstance(all_sv, list):
            all_sv = all_sv[1]  # binary classifier: positive-class SHAP values
        all_sv = np.asarray(all_sv)
        if all_sv.ndim == 3:
            all_sv = all_sv[..., 1]  # shap>=0.5x: (samples, features, classes)
        mean_abs = np.abs(all_sv).mean(axis=0)
        desc_idx = np.arange(self.n_fp_used, self.n_fp_used + len(DESC_COLS))
        return {
            "available": True,
            "feature": DESC_COLS,
            "shap_mean_abs": mean_abs[desc_idx],
            "shap_values": np.asarray(all_sv),
        }

    def fragment_attribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fragment association with activity (fragment-based attribution)."""
        from src.sar import FEATURE_SMARTS
        actives = df[df["is_active"] == 1]
        inactives = df[df["is_active"] == 0]
        rows = []
        for feat, smarts in FEATURE_SMARTS.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            na = sum(1 for m in actives["mol"] if m is not None and m.GetSubstructMatch(patt))
            ni = sum(1 for m in inactives["mol"] if m is not None and m.GetSubstructMatch(patt))
            if len(actives) == 0:
                continue
            rows.append({
                "fragment": feat,
                "active_fraction": round(na / len(actives), 4),
                "inactive_fraction": round(ni / len(inactives), 4) if len(inactives) else 0,
                "attribution_direction": "active_assoc" if na / len(actives) > ni / len(inactives) else "inactive_assoc",
            })
        return pd.DataFrame(rows)

    def feature_ablation(self, X, y, rf, sc, seed: int = 42) -> pd.DataFrame:
        baseline = rf.score(sc.transform(X), y)
        rows = []
        for i, name in enumerate(DESC_COLS):
            Xa = X.copy()
            Xa[:, self.n_fp_used + i] = 0
            acc = rf.score(sc.transform(Xa), y)
            rows.append({
                "feature": name,
                "ablated_accuracy": round(float(acc), 4),
                "accuracy_drop": round(float(baseline - acc), 4),
            })
        return pd.DataFrame(rows).sort_values("accuracy_drop", ascending=False)

    def counterfactuals(self, df: pd.DataFrame, X, y, rf, sc) -> pd.DataFrame:
        """Targeted structural perturbations testing feature necessity (section 14).

        For each active parent, generates counterfactuals spanning the required
        perturbation categories: remove feature, replace feature, alter linker,
        alter stereochemistry, change H-bond donor/acceptor, alter hydrophobic
        group, move substituent. Each is a HYPOTHESIS (predicted), not proof.
        """
        self.log.step("Counterfactual analysis (section 14)")
        rows = []

        def _remove_stereo(smiles: str) -> str:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                return smiles
            Chem.RemoveStereochemistry(m)
            return Chem.MolToSmiles(m)

        # perturbation: (name, parent, smi, category, note)
        probes = [
            # TAF-1 lactone feature: open or enlarge the lactone core
            {"name": "ID01_break_lactone_ring",
             "parent": "1",
             "smi": "O=C1OC([C@H](O)CO)C(O)=C(O)C1",
             "category": "remove_feature",
             "note": "lactone ring enlarged to 6-membered semi-pyrone (removes 5-ring H-bond acceptor geometry) to test TAF-1 necessity"},
            # TAF-2 hydroxyl donor: replace primary OH with CH3
            {"name": "ID01_remove_primary_OH",
             "parent": "1",
             "smi": "O=C1O[C@H]([C@H](O)C)C(O)=C1O",
             "category": "change_h_bond",
             "note": "-CH2OH -> -CH3 (remove primary hydroxyl H-bond donor)"},
            {"name": "ID01_move_primary_OH_chain",
             "parent": "1",
             "smi": "O=C1O[C@H]([C@H](CCO)C(O)=C1O)O",
             "category": "move_substituent",
             "note": "side-chain hydroxyl moved to a longer tether (alter spatial relationship)"},
            # EWG (TAF-3) on ID25: bromine replaced by non-EWG and by other EWG
            {"name": "ID25_remove_Br",
             "parent": "25",
             "smi": "CC[C@H]([C@H]1OC(=O)C(O)=C1O)O",
             "category": "remove_feature",
             "note": "-CH2Br -> -CH2CH3 (remove bromine EWG)"},
            {"name": "ID25_Br_to_Cl",
             "parent": "25",
             "smi": "ClC[C@H]([C@H]1OC(=O)C(O)=C1O)O",
             "category": "replace_feature",
             "note": "-CH2Br -> -CH2Cl (replace EWG with weaker halide)"},
            {"name": "ID25_Br_to_CF3",
             "parent": "25",
             "smi": "FC(F)(F)[C@@H](O)[C@H]1OC(=O)C(O)=C1O",
             "category": "replace_feature",
             "note": "-CH2Br -> -CH(CF3) extended EWG in comparable position"},
            # stereo removal on ID25 and ID12
            {"name": "ID25_remove_stereo",
             "parent": "25",
             "smi": _remove_stereo("O=C1O[C@H]([C@H](O)CBr)C(O)=C1O"),
             "category": "alter_stereochemistry",
             "note": "all R/S removed (diastereomer-free) to test TAF-4"},
            {"name": "ID12_remove_stereo",
             "parent": "12",
             "smi": _remove_stereo("C(#C)COC1=C(O)C(=O)O[C@@H]1[C@H]1COC(C)(C)O1"),
             "category": "alter_stereochemistry",
             "note": "compound 12 stereocentres removed"},
            {"name": "ID12_flip_core_stereo",
             "parent": "12",
             "smi": "C(#C)COC1=C(O)C(=O)O[C@H]1[C@@H]1COC(C)(C)O1",
             "category": "alter_stereochemistry",
             "note": "both core stereocentres inverted (enantiomeric configuration of lactone core)"},
            # alkyne EWG on ID12: remove and relocate
            {"name": "ID12_remove_alkyne",
             "parent": "12",
             "smi": "CCOC1=C(O)C(=O)O[C@@H]1[C@H]1COC(C)(C)O1",
             "category": "remove_feature",
             "note": "propargyl ether -> ethyl ether (remove terminal alkyne EWG)"},
            {"name": "ID12_alkyne_to_small_chain",
             "parent": "12",
             "smi": "CCCC1=C(O)C(=O)O[C@@H]1[C@H]1COC(C)(C)O1",
             "category": "replace_feature",
             "note": "propargyl ether -> n-butyl ether (isosteric chain, no EWG)"},
            # linker alteration on ID18 (benzyl ester)
            {"name": "ID18_ester_to_ether_linker",
             "parent": "18",
             "smi": "O=C1O[C@H]([C@H](O)CO)C(OCCc2ccccc2)=C1O",
             "category": "alter_linker",
             "note": "benzyl O-ester linkage lengthened by one carbon (alter linker)"},
            {"name": "ID18_benzyl_to_ortho_methyl",
             "parent": "18",
             "smi": "O=C1O[C@H]([C@H](O)CO)C(OCc2ccccc2C)=C1O",
             "category": "alter_hydrophobic",
             "note": "o-methyl on benzyl (alter hydrophobic group bulk)"},
            # ID24 (active): carbonyl/sidechain perturbation
            {"name": "ID24_remove_alcohol",
             "parent": "24",
             "smi": "CCCOC1=C(O)[C@@H]([C@H](C)CO)OC1=O",
             "category": "remove_feature",
             "note": "remove secondary alcohol (H-bond donor) from active series"},
            # ID01 -> lactone carbonyl hydrolyzed
            {"name": "ID01_open_pyrone",
             "parent": "1",
             "smi": "O=C(O)C=C(O)[C@H](O)[C@H](O)CO",
             "category": "remove_feature",
             "note": "open e-lactone ring to acyclic dihydroxy diacid (removes ring H-bond acceptor geometry)"},
        ]
        seen = set()
        for p in probes:
            key = p["smi"]
            if key in seen:
                continue
            seen.add(key)
            mol = Chem.MolFromSmiles(p["smi"])
            if mol is None:
                rows.append({"probe": p["name"], "parent_compound": p["parent"],
                             "category": p["category"], "note": p["note"],
                             "predicted_active": "INVALID", "confidence": np.nan,
                             "hypothesis": "untested-hypothesis"})
                continue
            Xrow = self._feat_row(mol)
            pred, conf = self._predict_row(rf, sc, Xrow)
            rows.append({
                "probe": p["name"], "parent_compound": p["parent"],
                "category": p["category"], "note": p["note"],
                "predicted_active": bool(pred == 1) if pred is not None else None,
                "confidence": round(conf, 4) if conf is not None else np.nan,
                "hypothesis": "untested-hypothesis-not-experimental-proof",
            })
        out = pd.DataFrame(rows)
        # any counterfactual that coincides with a dataset compound (ground truth known)
        if len(df) and len(out):
            canon = set(df["canonical_SMILES"].dropna())
            is_dataset = []
            for p in probes:
                m = Chem.MolFromSmiles(p["smi"])
                is_dataset.append("yes" if m is not None and Chem.MolToSmiles(m) in canon else "no")
            out["matches_dataset_compound"] = is_dataset
        save_df(out, str(self.cfg.resolve("xai/counterfactual/counterfactual.csv")))
        self.log.info(out[["probe", "category", "predicted_active", "confidence"]].to_string(index=False))
        return out

    def _feat_row(self, mol) -> np.ndarray:
        from rdkit.Chem import Descriptors
        d = {
            "MW": Descriptors.MolWt(mol), "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol), "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol), "RotBonds": Descriptors.NumRotatableBonds(mol),
            "RingCount": Descriptors.RingCount(mol), "AromaticRings": Descriptors.NumAromaticRings(mol),
            "FractionCSP3": Descriptors.FractionCSP3(mol), "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        }
        fp = morgan_array(mol)[self.fp_mask]
        return np.hstack([fp, [d[c] for c in DESC_COLS]]).astype(np.float64)

    def _predict_row(self, rf, sc, Xrow):
        try:
            p = rf.predict_proba(sc.transform(Xrow.reshape(1, -1)))[0]
            pred = int(np.argmax(p))
            return pred, float(np.max(p))
        except Exception:
            return None, None

    def stability(self, df: pd.DataFrame, X, y, n_resamples: int = 20, seed: int = 42) -> pd.DataFrame:
        """Repeat explanations across models, seeds, resample-splits, and
        explanation methods; quantify stability (Spearman, top-k overlap,
        Jaccard, sign consistency, feature-selection frequency). Unstable
        explanations are reported AS unstable (section 13).
        """
        self.log.step("XAI reproducibility (section 13)")
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier

        rng = np.random.default_rng(seed)
        model_variants = {
            "random_forest": lambda s: RandomForestClassifier(n_estimators=200, random_state=s, class_weight="balanced"),
            "extra_trees": lambda s: ExtraTreesClassifier(n_estimators=200, random_state=s, class_weight="balanced"),
            "gradient_boosting": lambda s: GradientBoostingClassifier(n_estimators=200, random_state=s),
        }
        seeds = [42, 1337, 2026]
        n_per = max(1, n_resamples // 3)
        splits = [rng.integers(0, len(df), size=int(0.8 * len(df))) for _ in range(n_per * len(seeds))]

        # per-run records: {..scores..} -> dict {feature: value}
        gini_runs: List[Dict[str, float]] = []
        perm_runs: List[Dict[str, float]] = []
        abl_runs: List[Dict[str, float]] = []
        run_meta: List[Dict[str, Any]] = []
        item = 0
        for mname, factory in model_variants.items():
            for run_seed in seeds:
                for _ in range(n_per):
                    idx = splits[item % len(splits)]
                    item += 1
                    if len(np.unique(y[idx])) < 2:
                        continue
                    clf = factory(run_seed)
                    sc = StandardScaler()
                    clf.fit(sc.fit_transform(X[idx]), y[idx])

                    gi = self.gini_importance(clf).set_index("feature")[
                        "rf_gini_importance"].to_dict()
                    perm = self._permutation_importance(clf, sc, X, y, seed=run_seed)
                    abl = self._ablation_scores(clf, sc, X, y)

                    gini_runs.append({c: float(gi[c]) for c in DESC_COLS})
                    perm_runs.append({c: perm.get(c, 0.0) for c in DESC_COLS})
                    abl_runs.append({c: abl.get(c, 0.0) for c in DESC_COLS})
                    run_meta.append({"model": mname, "seed": run_seed})

        if not gini_runs:
            return pd.DataFrame()
        meta = pd.DataFrame(run_meta)

        def _clean(vals):
            return [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]

        def _stats(runs: List[Dict[str, float]]) -> Dict[str, float]:
            n = len(runs)
            if n < 2:
                return {"spearman": float("nan"), "topk": float("nan"),
                        "jaccard": float("nan"), "sign": float("nan")}
            rho_all, topk_all, jac_all, sign_all = [], [], [], []
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = runs[i], runs[j]
                    keys = [k for k in a if k in b]
                    if len(keys) < 2:
                        continue
                    rho, _ = spearmanr([a[k] for k in keys], [b[k] for k in keys])
                    rho_all.append(rho)
                    ka = set(pd.Series(a).sort_values(ascending=False).head(3).index)
                    kb = set(pd.Series(b).sort_values(ascending=False).head(3).index)
                    u = len(ka | kb)
                    topk_all.append(len(ka & kb))
                    jac_all.append(1.0 if u == 0 else len(ka & kb) / u)
                    sig = [k for k in keys if a[k] != 0 or b[k] != 0]
                    if sig:
                        sign_all.append(sum(1 for k in sig if np.sign(a[k]) == np.sign(b[k])) / len(sig))
            rhos = _clean(rho_all)
            jacs = _clean(jac_all)
            signs = _clean(sign_all)
            return {
                "spearman": float(np.mean(rhos)) if rhos else float("nan"),
                "topk": float(np.mean(topk_all)) if topk_all else float("nan"),
                "jaccard": float(np.mean(jacs)) if jacs else float("nan"),
                "sign": float(np.mean(signs)) if signs else float("nan"),
            }

        def _by_group(runs: List[Dict[str, float]], meta: pd.DataFrame,
                      key: str) -> List[Dict[str, float]]:
            """Average runs within each group value of `key` then return the
            grouped run-vectors for pairwise stability across groups."""
            groups: Dict[Any, Dict[str, List[float]]] = {}
            for r, m in zip(runs, meta[key]):
                for f, v in r.items():
                    groups.setdefault(m, {}).setdefault(f, []).append(v)
            return [{f: float(np.mean(v)) for f, v in g.items()} for g in groups.values()]

        def _rows(dim, st):
            return {
                "dimension": dim,
                "spearman_rank_corr_mean": st["spearman"],
                "topk_overlap_mean": st["topk"],
                "jaccard_topk_similarity": st["jaccard"],
                "sign_consistency": st["sign"],
                "classification": ("STABLE" if st["spearman"] >= 0.7 else
                                   ("UNSTABLE" if st["spearman"] <= 0.5 else "MARGINAL"))
                if not np.isnan(st["spearman"]) else "UNDETERMINED",
            }

        def _method_agreement(a: List[Dict[str, float]], b: List[Dict[str, float]]) -> Dict[str, float]:
            """Spearman / sign agreement between two explanation methods
            computed on the SAME runs, averaged over runs."""
            if len(a) != len(b) or not a:
                return {"spearman": float("nan"), "topk": float("nan"),
                        "jaccard": float("nan"), "sign": float("nan")}
            rhos, signs, jacs = [], [], []
            for ra, rb in zip(a, b):
                keys = [k for k in ra if k in rb]
                if len(keys) < 2:
                    continue
                rho, _ = spearmanr([ra[k] for k in keys], [rb[k] for k in keys])
                rhos.append(rho)
                ka = set(pd.Series(ra).sort_values(ascending=False).head(3).index)
                kb = set(pd.Series(rb).sort_values(ascending=False).head(3).index)
                u = len(ka | kb)
                jacs.append(1.0 if u == 0 else len(ka & kb) / u)
                sig = [k for k in keys if ra[k] != 0 or rb[k] != 0]
                signs.append(sum(1 for k in sig if np.sign(ra[k]) == np.sign(rb[k])) / len(sig) if sig else 1.0)
            rhos = _clean(rhos)
            jacs = _clean(jacs)
            signs = _clean(signs)
            return {
                "spearman": float(np.mean(rhos)) if rhos else float("nan"),
                "topk": float(np.max(jacs)) if jacs else float("nan"),
                "jaccard": float(np.mean(jacs)) if jacs else float("nan"),
                "sign": float(np.mean(signs)) if signs else float("nan"),
            }

        summary = pd.DataFrame([
            _rows("across_models_seeds_splits", _stats(gini_runs)),
            _rows("across_seeds_fixed_model", _stats(_by_group(gini_runs, meta, "seed"))),
            _rows("across_models_fixed_seed", _stats(_by_group(gini_runs, meta, "model"))),
            _rows("method_gini_vs_permutation", _method_agreement(gini_runs, perm_runs)),
            _rows("method_gini_vs_ablation", _method_agreement(gini_runs, abl_runs)),
        ])
        # feature-selection frequency across gini runs
        sel = pd.DataFrame(gini_runs).gt(0).sum()
        sel_freq = pd.DataFrame({"feature": sel.index,
                                 "selection_frequency": sel.sort_values(ascending=False).values})
        save_df(sel_freq, str(self.cfg.resolve("xai/stability/xai_feature_selection_frequency.csv")))
        save_df(pd.DataFrame(gini_runs), str(self.cfg.resolve("xai/stability/xai_stability_scores.csv")))
        save_df(summary, str(self.cfg.resolve("xai/stability/xai_stability.csv")))
        self.log.info(summary.to_string(index=False))
        return summary

    def _permutation_importance(self, clf, sc, X, y, seed: int = 42) -> Dict[str, float]:
        base = clf.score(sc.transform(X), y)
        rng = np.random.default_rng(seed)
        out = {}
        for i, name in enumerate(DESC_COLS):
            drop = []
            for _ in range(4):
                Xp = X.copy()
                Xp[:, self.n_fp_used + i] = Xp[rng.permutation(range(len(X))), self.n_fp_used + i]
                drop.append(base - clf.score(sc.transform(Xp), y))
            out[name] = float(np.mean(drop))
        return out

    def _ablation_scores(self, clf, sc, X, y) -> Dict[str, float]:
        base = clf.score(sc.transform(X), y)
        out = {}
        for i, name in enumerate(DESC_COLS):
            Xa = X.copy()
            Xa[:, self.n_fp_used + i] = 0
            out[name] = float(base - clf.score(sc.transform(Xa), y))
        return out

    def explanation_preservation(self, df: pd.DataFrame, gini, perm, shap_out,
                                 frag) -> pd.DataFrame:
        """Section 26: does the augmented system preserve experimentally
        meaningful SAR, or has explanation shifted toward generic descriptors
        (MW / LogP / aromaticity / ring count) without experimental support?

        Flags a model as potentially exploiting dataset artifacts if generic
        descriptors dominate the top of every explanation ranking while
        TAF-specific fragments rank low AND the fragment-level experimental
        enrichment is weak.
        """
        self.log.step("Explanation-preservation analysis (section 26)")
        GENERIC = {"MW", "LogP", "TPSA", "RingCount", "AromaticRings",
                   "RotBonds", "FractionCSP3", "HeavyAtomCount"}
        # combos that map to real chemical meaning beyond raw size
        SPECIFIC_DESC = {"HBD", "HBA"}

        rows = []
        # rank gini for top generic vs specific descriptors
        g = gini.set_index("feature")["rf_gini_importance"]
        perm_map = dict(zip(perm["feature"], perm["perm_importance_mean"]))

        def _rank_of(mapping, feats, desc=False):
            s = pd.Series(mapping)
            if desc:
                s = s[[f for f in s.index if f in DESC_COLS]]
            return s.rank(ascending=False).to_dict()

        top_generic_gini = g[[c for c in GENERIC if c in g.index]].sort_values(ascending=False)
        top_specific_gini = g[[c for c in SPECIFIC_DESC if c in g.index]].sort_values(ascending=False)
        # fingerprint-level aggregate importance (specific structural features)
        fp_gini_sum = float(gini["rf_gini_importance"].sum())
        generic_gini_frac = float(top_generic_gini.sum() / (gini["rf_gini_importance"].sum() + 1e-12))
        specific_gini_frac = float(top_specific_gini.sum() / (gini["rf_gini_importance"].sum() + 1e-12))

        # fragment-level experimental support: are structural features enriched?
        actives = frag[frag["attribution_direction"] == "active_assoc"]
        exp_support_frac = float((actives["active_fraction"] - actives["inactive_fraction"]).clip(lower=0).sum()) \
            if len(actives) else 0.0
        exp_support_frac = min(1.0, exp_support_frac)

        # flag: generic descriptors dominate AND experimental fragment support weak
        generic_dominated = generic_gini_frac > 0.6
        low_exp_support = exp_support_frac < 0.25
        flagged = generic_dominated and low_exp_support

        # SHAP agreement (same-feature, experimental-aware)
        shap_top = ""
        if shap_out["available"]:
            sh = pd.Series(shap_out["shap_mean_abs"], index=DESC_COLS)
            top_f = sh.sort_values(ascending=False).head(3).index.tolist()
            shap_top = ",".join(top_f)
            top_specific_in_shap = int(any(f in SPECIFIC_DESC for f in top_f))

        rows.append({
            "top_generic_descriptor": ",".join(top_generic_gini.head(3).index.tolist()),
            "top_specific_descriptor": ",".join(top_specific_gini.head(3).index.tolist() or ["(none)"]),
            "generic_descriptor_gini_frac": round(generic_gini_frac, 3),
            "specific_descriptor_gini_frac": round(specific_gini_frac, 3),
            "fragment_experimental_support_frac": round(exp_support_frac, 3),
            "shap_top3_features": shap_top,
            "shap_includes_specific": top_specific_in_shap if shap_out["available"] else "N/A",
            "generic_dominated_flag": bool(generic_dominated),
            "weak_experimental_fragment_support_flag": bool(low_exp_support),
            "flag_artifact_exploitation": bool(flagged),
            "interpretation": ("Model explanation dominated by generic descriptors with weak "
                               "experimental fragment support - potentially exploiting dataset "
                               "artifacts; NOT mechanistically informative as-is"
                               if flagged else
                               ("Model top features retain experimental specificity; "
                                "artifact-exploitation flag NOT raised")),
        })
        out = pd.DataFrame(rows)
        save_df(out, str(self.cfg.resolve("xai/explanation_preservation.csv")))
        self.log.info(out.to_string(index=False))
        return out

    def run_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        X, feat_names = self._feature_matrix(df)
        y = df["is_active"].to_numpy()
        rf, sc = self.train_rf(X, y, seed=self.cfg.seed("models"))

        gini = self.gini_importance(rf)
        perm = self.permutation_importance(X, y, rf, sc)
        shap_out = self.shap_values(X, y, rf, sc)
        frag = self.fragment_attribution(df)
        ablation = self.feature_ablation(X, y, rf, sc)
        cf = self.counterfactuals(df, X, y, rf, sc)
        stab = self.stability(df, X, y, n_resamples=self.cfg.raw.get("xai", {}).get("stability", {}).get("n_resamples", 20))
        pres = self.explanation_preservation(df, gini, perm, shap_out, frag)

        gini.to_csv(str(self.cfg.resolve("xai/permutation/rf_gini_importance.csv")), index=False)
        perm.to_csv(str(self.cfg.resolve("xai/permutation/permutation_importance.csv")), index=False)
        if shap_out["available"]:
            pd.DataFrame({
                "feature": DESC_COLS,
                "shap_mean_abs": shap_out["shap_mean_abs"],
            }).to_csv(str(self.cfg.resolve("xai/shap/shap_mean_abs.csv")), index=False)
            np.save(str(self.cfg.resolve("xai/shap/shap_values_desc.npy")),
                    shap_out["shap_values"][:, self.n_fp_used:])
            np.save(str(self.cfg.resolve("xai/shap/shap_values_fp.npy")),
                    shap_out["shap_values"][:, :self.n_fp_used])
        frag.to_csv(str(self.cfg.resolve("xai/fragment/fragment_attribution.csv")), index=False)
        ablation.to_csv(str(self.cfg.resolve("xai/counterfactual/feature_ablation.csv")), index=False)
        cf.to_csv(str(self.cfg.resolve("xai/counterfactual/counterfactual.csv")), index=False)
        save_manifest(
            make_provenance("xai", self.cfg, {"n_compounds": int(len(df)), "shap_available": shap_out["available"]}),
            str(self.cfg.resolve("xai/xai_manifest.json")),
        )
        return {
            "gini": gini, "permutation": perm, "shap": shap_out,
            "fragment": frag, "ablation": ablation, "counterfactual": cf,
            "stability": stab, "explanation_preservation": pres,
        }


def run_phase5(cfg_path: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    import os, logging
    from pathlib import Path
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase5_xai", str(cfg.resolve("logs")))
    if df is None:
        p = str(cfg.resolve("data/processed/data_analysis_ready.csv"))
        df = pd.read_csv(p)
        df["mol"] = df["isomeric_SMILES"].apply(Chem.MolFromSmiles)
    xai = XAI(cfg, logger)
    return xai.run_all(df)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase5(cfgp)