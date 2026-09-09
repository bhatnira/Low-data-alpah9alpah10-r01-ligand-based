#!/usr/bin/env python3
"""Phase 4: Baseline models and rigorous validation.

Implements prompt2.txt sections 8, 9, 10, 11, 31 (statistical plan) and
report 3:
  - model panel: majority/random baselines + classical interpretable models
  - representations: descriptors, Morgan FP, descriptors+FP
  - primary validation: LOCO, leave-one-series-out, scaffold-out
  - secondary validation: chemical-distance, bootstrap, permutation test
  - applicability domain (IN / NEAR / OUT) + separate uncertainty axes
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
    sha256_file,
)

RDLogger.DisableLog("rdApp.*")

DESC_COLS = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
             "RingCount", "AromaticRings", "FractionCSP3", "HeavyAtomCount"]


def morgan_array(mol, radius=2, nbits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class FeatureSet:
    """Bundled feature matrix with a documented representation."""

    def __init__(self, X: np.ndarray, feature_names: List[str], representation: str):
        self.X = X
        self.feature_names = feature_names
        self.representation = representation


class Representations:
    def build(self, df: pd.DataFrame) -> Dict[str, FeatureSet]:
        mols = df["mol"].tolist()
        desc = df[DESC_COLS].fillna(0).to_numpy(dtype=np.float64)
        fps = np.vstack([morgan_array(m) if m is not None else np.zeros(2048) for m in mols])
        fps = fps.astype(np.float64)
        fp_names = [f"morgan_{i}" for i in range(2048)]
        return {
            "descriptors_only": FeatureSet(desc, DESC_COLS, "descriptors_only"),
            "fingerprints_morgan_r2": FeatureSet(fps, fp_names, "fingerprints_morgan_r2"),
            "fingerprints_plus_descriptors": FeatureSet(
                np.hstack([fps, desc]),
                fp_names + DESC_COLS,
                "fingerprints_plus_descriptors",
            ),
        }


def _build_models(cfg: ProjectConfig) -> List[Tuple[str, Any]]:
    seed = cfg.seed("models")
    return [
        ("majority", DummyClassifier(strategy="most_frequent", random_state=seed)),
        ("random", DummyClassifier(strategy="stratified", random_state=seed)),
        ("logreg_desc", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ("ridge", RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=seed)),
        ("random_forest", RandomForestClassifier(n_estimators=100, random_state=seed, class_weight="balanced")),
        ("extra_trees", ExtraTreesClassifier(n_estimators=100, random_state=seed, class_weight="balanced")),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=50, random_state=seed)),
    ]


class Validator:
    """Handles the several validation strategies with dataset == ground truth."""

    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger

    def _pred_loop(self, model, X, y, train_idx, test_idx, probability=False):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[train_idx])
        Xte = sc.transform(X[test_idx])
        model.fit(Xtr, y[train_idx])
        if probability and hasattr(model, "predict_proba"):
            return model.predict_proba(Xte)[:, 1]
        return model.predict(Xte)

    def loco(self, feats: Dict[str, FeatureSet], df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        self.log.step("Leave-one-compound-out (LOCO)")
        y = df["is_active"].to_numpy()
        results = []
        per_model = {name: [] for name, _ in _build_models(self.cfg)}
        for name, model in _build_models(self.cfg):
            for rep, fs in feats.items():
                for test_idx in range(len(df)):
                    train = np.array([k for k in range(len(df)) if k != test_idx])
                    pred = self._pred_loop(model, fs.X, y, train, np.array([test_idx]))
                    results.append({
                        "model": name, "representation": rep,
                        "compound_id": int(df.iloc[test_idx]["Identifier"]),
                        "true": int(y[test_idx]), "pred": int(pred[0]),
                    })
        res = pd.DataFrame(results)
        summary = res.groupby(["model", "representation"]).apply(
            lambda g: pd.Series({
                "n": len(g),
                "n_correct": int((g["true"] == g["pred"]).sum()),
                "accuracy": round(float((g["true"] == g["pred"]).mean()), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(g["true"], g["pred"])), 4),
            }), include_groups=False
        ).reset_index()
        save_df(res, str(self.cfg.resolve("validation/loco/loco_predictions.csv")))
        save_df(summary, str(self.cfg.resolve("validation/loco/loco_summary.csv")))
        self.log.info(summary.to_string(index=False))
        return {"predictions": res, "summary": summary}

    def series_and_scaffold_out(self, feats: Dict[str, FeatureSet], df: pd.DataFrame,
                                by: str = "murcko_scaffold") -> Dict[str, pd.DataFrame]:
        self.log.step(f"Leave-{by}-out")
        y = df["is_active"].to_numpy()
        groups = df[by].fillna("NO_SCAFFOLD").tolist()
        unique = [g for g in sorted(set(groups)) if g is not None]
        results = []
        for name, model in _build_models(self.cfg):
            for rep, fs in feats.items():
                for g in unique:
                    test_idx = np.array([k for k in range(len(df)) if groups[k] == g])
                    train_idx = np.array([k for k in range(len(df)) if groups[k] != g])
                    if len(np.unique(y[train_idx])) < 2:
                        continue
                    pred = self._pred_loop(model, fs.X, y, train_idx, test_idx)
                    for k, t in enumerate(test_idx):
                        results.append({
                            "strategy": by, "model": name, "representation": rep,
                            "held_out_group": str(g),
                            "compound_id": int(df.iloc[t]["Identifier"]),
                            "true": int(y[t]), "pred": int(pred[k]),
                        })
        res = pd.DataFrame(results)
        out_name = "series_out" if by != "murcko_scaffold" else "scaffold_out"
        save_df(res, str(self.cfg.resolve(f"validation/{out_name}/{out_name}_predictions.csv")))
        if len(res):
            summary = res.groupby(["strategy", "model", "representation"]).apply(
                lambda g: pd.Series({
                    "n": len(g),
                    "n_correct": int((g["true"] == g["pred"]).sum()),
                    "accuracy": round(float((g["true"] == g["pred"]).mean()), 4),
                }), include_groups=False
            ).reset_index()
            save_df(summary, str(self.cfg.resolve(f"validation/{out_name}/{out_name}_summary.csv")))
            self.log.info(f"scaffold-out summary (accuracy): "
                          f"{dict(summary.drop_duplicates(subset=['model','representation']).set_index('model')['accuracy'])}")
        return {"predictions": res}

    def series_out(self, feats: Dict[str, FeatureSet], df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Leave-one-series-out: train on all but one analog series and test on the
        held-out series (section 9).  With n=30 and 17 unique scaffolds the series
        are nearly singletons, so this approximates scaffold-out; the result is
        reported as such rather than hidden."""
        self.log.step("Leave-one-series-out")
        y = df["is_active"].to_numpy()
        groups = df["series_id"].fillna(-1).tolist()
        unique = sorted({g for g in groups if g != -1})
        results = []
        for name, model in _build_models(self.cfg):
            for rep, fs in feats.items():
                for g in unique:
                    test_idx = np.array([k for k in range(len(df)) if groups[k] == g])
                    train_idx = np.array([k for k in range(len(df)) if groups[k] != g])
                    if len(np.unique(y[train_idx])) < 2:
                        continue
                    pred = self._pred_loop(model, fs.X, y, train_idx, test_idx)
                    for k, t in enumerate(test_idx):
                        results.append({
                            "strategy": "series_out", "model": name, "representation": rep,
                            "held_out_group": str(g),
                            "compound_id": int(df.iloc[t]["Identifier"]),
                            "true": int(y[t]), "pred": int(pred[k]),
                        })
        res = pd.DataFrame(results)
        save_df(res, str(self.cfg.resolve("validation/series_out/series_out_predictions.csv")))
        if len(res):
            summary = res.groupby(["strategy", "model", "representation"]).apply(
                lambda g: pd.Series({
                    "n": len(g),
                    "n_correct": int((g["true"] == g["pred"]).sum()),
                    "accuracy": round(float((g["true"] == g["pred"]).mean()), 4),
                    "n_positive": int(g["true"].sum()),
                }), include_groups=False
            ).reset_index()
            save_df(summary, str(self.cfg.resolve("validation/series_out/series_out_summary.csv")))
            self.log.info(f"series-out summary written (n series = {len(unique)})")
        else:
            self.log.warn("series-out produced no rows")
        return {"predictions": res}

    def chemical_distance(self, feats: Dict[str, FeatureSet], df: pd.DataFrame,
                          sim_matrix: np.ndarray) -> pd.DataFrame:
        """Evaluate performance as chemical distance from each training point grows."""
        self.log.step("Chemical-distance validation")
        y = df["is_active"].to_numpy()
        # nearest-neighbour distance to training set (leave-one-out style)
        nn_dist = []
        for i in range(len(df)):
            d = 1 - sim_matrix[i, :]
            mask = np.arange(len(df)) != i
            nn_dist.append(np.min(d[mask]))
        dist_col = np.array(nn_dist)
        bins = pd.qcut(dist_col, q=min(4, len(np.unique(dist_col))), duplicates="drop")
        results = []
        for name, model in _build_models(self.cfg):
            for rep, fs in feats.items():
                for bi, b in enumerate(pd.unique(bins)):
                    test_idx = np.where(bins == b)[0]
                    train_idx = np.array([k for k in range(len(df)) if k not in test_idx])
                    if len(np.unique(y[train_idx])) < 2:
                        continue
                    pred = self._pred_loop(model, fs.X, y, train_idx, test_idx)
                    results.append({
                        "model": name, "representation": rep,
                        "bin": str(b), "n": len(test_idx),
                        "accuracy": round(float((pred == y[test_idx]).mean()), 4),
                        "mean_nn_distance": round(float(dist_col[test_idx].mean()), 4),
                        "level_hint": self._classify_distance(dist_col[test_idx].mean()),
                    })
        res = pd.DataFrame(results)
        save_df(res, str(self.cfg.resolve("validation/chemical_distance/chemical_distance.csv")))
        return res

    @staticmethod
    def _classify_distance(d: float) -> str:
        if d < 0.25:
            return "same_scaffold"
        if d < 0.5:
            return "close_analogue"
        if d < 0.7:
            return "bioisostere"
        if d < 0.8:
            return "moderate_hop"
        if d < 0.9:
            return "distant_scaffold"
        return "novel_scaffold"

    def bootstrap(self, feats: Dict[str, FeatureSet], df: pd.DataFrame,
                  n_resamples: int = 1000, seed: int = 42) -> pd.DataFrame:
        self.log.step("Bootstrap performance uncertainty")
        rng = np.random.default_rng(seed)
        y = df["is_active"].to_numpy()
        rows = []
        for name, model in _build_models(self.cfg):
            if name in ("majority", "random"):
                continue
            fs = feats["fingerprints_plus_descriptors"]
            for r in range(n_resamples):
                idx = rng.integers(0, len(df), size=len(df))
                if len(np.unique(y[idx])) < 2:
                    continue
                try:
                    pred = self._pred_loop(model, fs.X, y, idx, np.arange(len(df)))
                    rows.append({"model": name, "resample": r,
                                 "accuracy": float((pred == y).mean())})
                except Exception:
                    continue
        res = pd.DataFrame(rows)
        summary = res.groupby("model")["accuracy"].agg(["mean", "std", "min", "max"]).round(4).reset_index()
        summary.columns = ["model", "mean", "std", "ci_low", "ci_high"]
        rng2 = np.random.default_rng(seed)
        for model in summary["model"]:
            vals = res.loc[res["model"] == model, "accuracy"].to_numpy()
            if len(vals):
                low, high = np.percentile(vals, [2.5, 97.5])
                summary.loc[summary["model"] == model, "ci_low"] = round(low, 4)
                summary.loc[summary["model"] == model, "ci_high"] = round(high, 4)
        save_df(summary, str(self.cfg.resolve("validation/bootstrap/bootstrap_summary.csv")))
        self.log.info(summary.to_string(index=False))
        return summary

    def permutation_test(self, feats: Dict[str, FeatureSet], df: pd.DataFrame,
                         n_perm: int = 200, seed: int = 42) -> pd.DataFrame:
        self.log.step("Permutation test (chance-level null)")
        y0 = df["is_active"].to_numpy()
        rng = np.random.default_rng(seed)
        fs = feats["fingerprints_plus_descriptors"]
        model = RandomForestClassifier(n_estimators=100, random_state=self.cfg.seed("models"),
                                       class_weight="balanced")
        idx = np.arange(len(df))
        sc = StandardScaler()
        obs = self._pred_loop(model, fs.X, y0, idx, idx)
        obs_acc = float((obs == y0).mean())
        perm_accs = []
        for _ in range(n_perm):
            yp = rng.permutation(y0)
            p = self._pred_loop(model, fs.X, yp, idx, idx)
            perm_accs.append(float((p == yp).mean()))
        perm_accs = np.array(perm_accs)
        p_value = float((perm_accs >= obs_acc).mean())
        res = pd.DataFrame({
            "model": ["random_forest"] * len(perm_accs) + ["random_forest"],
            "kind": ["permuted"] * len(perm_accs) + ["observed"],
            "accuracy": list(perm_accs) + [obs_acc],
        })
        summary = pd.DataFrame([{
            "model": "random_forest",
            "observed_accuracy": round(obs_acc, 4),
            "perm_mean": round(float(perm_accs.mean()), 4),
            "perm_std": round(float(perm_accs.std()), 4),
            "perm_p_value": round(p_value, 4),
            "n_permutations": n_perm,
        }])
        save_df(res, str(self.cfg.resolve("validation/permutation/permutation_scores.csv")))
        save_df(summary, str(self.cfg.resolve("validation/permutation/permutation_summary.csv")))
        self.log.info(summary.to_string(index=False))
        return summary

    def applicability_domain(self, df: pd.DataFrame, sim_matrix: np.ndarray) -> pd.DataFrame:
        """Threshold distance-based AD: IN/NEAR/OUT based on training-set coverage."""
        self.log.step("Applicability domain (section 10)")
        y = df["is_active"].to_numpy()
        active_ids = np.where(y == 1)[0]
        rows = []
        for i in range(len(df)):
            d = 1 - sim_matrix[i, :]
            d_active = float(d[active_ids].min() if len(active_ids) else np.nan)
            d_any = float(d.min())
            med_active = float(np.median(d[active_ids])) if len(active_ids) else np.nan
            if d_any < 0.35:
                ad = "IN-DOMAIN"
            elif d_any < 0.6 or d_active < 0.45:
                ad = "NEAR-DOMAIN"
            else:
                ad = "OUT-OF-DOMAIN"
            rows.append({
                "compound_id": int(df.iloc[i]["Identifier"]),
                "min_distance_to_any": round(d_any, 4),
                "min_distance_to_active": round(d_active, 4) if not np.isnan(d_active) else np.nan,
                "median_distance_to_active": round(med_active, 4) if not np.isnan(med_active) else np.nan,
                "applicability_domain": ad,
                "is_active": int(y[i]),
            })
        res = pd.DataFrame(rows)
        save_df(res, str(self.cfg.resolve("validation/applicability_domain.csv")))
        self.log.info(res["applicability_domain"].value_counts().to_string())
        return res

    def uncertainty_axes(self, feats: Dict[str, FeatureSet], df: pd.DataFrame,
                         sim_matrix: np.ndarray) -> pd.DataFrame:
        """Separate predictive vs novelty vs AD uncertainty (section 11).

        Three independent axes, never collapsed into one number:
          * predictive_uncertainty - ensemble variance (tree-level std of the
            predicted positive-class probability) across the frozen RF
          * model_disagreement     - std of the predicted probability across the
            model panel (RF / ExtraTrees / GBM), capturing hypothesis disagreement
          * chemical_novelty       - min distance to training set (fingerprint)
          * ad_uncertainty         - distance to nearest active scaled to [0,1]
        """
        self.log.step("Uncertainty axes (section 11)")
        y = df["is_active"].to_numpy()
        active_ids = np.where(y == 1)[0]
        fs = feats["fingerprints_plus_descriptors"]
        X = fs.X.astype(np.float64)
        sc = StandardScaler()
        Xs = sc.fit_transform(X)

        seed = self.cfg.seed("models")
        rf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
        rf.fit(Xs, y)
        tree_probas = np.stack([t.predict_proba(Xs)[:, 1] for t in rf.estimators_])
        predictive_unc = tree_probas.std(axis=0)
        rf_proba = tree_probas.mean(axis=0)

        et = ExtraTreesClassifier(n_estimators=100, random_state=seed, class_weight="balanced").fit(Xs, y)
        gbm = GradientBoostingClassifier(n_estimators=50, random_state=seed).fit(Xs, y)
        probas = [rf_proba, et.predict_proba(Xs)[:, 1], gbm.predict_proba(Xs)[:, 1]]
        model_disagreement = np.std(probas, axis=0)

        rows = []
        for i in range(len(df)):
            d = 1 - sim_matrix[i, :]
            d_any = float(d.min())
            d_active = float(d[active_ids].min()) if len(active_ids) else np.nan
            rows.append({
                "compound_id": int(df.iloc[i]["Identifier"]),
                "predictive_uncertainty": round(float(predictive_unc[i]), 4),
                "model_disagreement": round(float(model_disagreement[i]), 4),
                "predictive_probability": round(float(rf_proba[i]), 4),
                "chemical_novelty": round(d_any, 4),
                "ad_uncertainty": round(float(np.clip(d_active - 0.35, 0, 1)), 4)
                if not np.isnan(d_active) else np.nan,
                "is_active": int(y[i]),
            })
        res = pd.DataFrame(rows)
        save_df(res, str(self.cfg.resolve("validation/uncertainty_axes.csv")))
        self.log.info(res[["compound_id", "predictive_uncertainty",
                           "model_disagreement", "ad_uncertainty"]].to_string(index=False))
        return res

    def export_frozen_model(self, reps: Dict[str, FeatureSet], df: pd.DataFrame):
        """Export the frozen classical model artifact used by the REINVENT 4
        scoring component (reinvent/score_lbm.py via ExternalProcess).

        The model is trained once on the full, locked analysis dataset, before
        any prospective generation.  No generated molecule is ever used for
        fitting, so the prospective score is consistent with the model lock
        manifest and no retraining information leaks into generation.
        """
        self.log.step("Export frozen model artifact for prospective scoring")
        fs = reps["fingerprints_plus_descriptors"]
        X = fs.X.astype(np.float64)
        y = df["is_active"].to_numpy(dtype=int)
        sc = StandardScaler().fit(X)
        clf = RandomForestClassifier(n_estimators=200, random_state=self.cfg.seed("models"),
                                     class_weight="balanced")
        clf.fit(sc.transform(X), y)
        act = df[df["is_active"] == 1]
        artifact = {
            "version": 1,
            "description": (
                "Frozen classical RF classifier (Morgan r=2 nbits=2048 + RDKit "
                "descriptors) trained once on the locked analysis dataset. "
                "Consumed by REINVENT4 ExternalProcess scoring component "
                "(reinvent/score_lbm.py)."
            ),
            "model": clf,
            "scaler": sc,
            "representation": fs.representation,
            "feature_columns": fs.feature_names,
            "desc_columns": list(DESC_COLS),
            "nbits": 2048,
            "radius": 2,
            "seed": self.cfg.seed("models"),
            "n_compounds": int(len(df)),
            "n_actives": int(y.sum()),
            "positive_class": 1,
            "training_smiles": df["isomeric_SMILES"].tolist(),
            "active_smiles": act["isomeric_SMILES"].tolist(),
            "training_murcko": sorted({str(s) for s in df["murcko_scaffold"].dropna().tolist()}),
            "training_data_sha256": sha256_file(
                str(self.cfg.resolve("data/processed/data_analysis_ready.csv"))
            ),
        }
        out = self.cfg.resolve("models/classical/lbm_rf_fp_desc.joblib")
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, out)
        save_manifest(
            make_provenance("models/frozen", self.cfg, {
                "artifact": str(out),
                "representation": fs.representation,
                "n_compounds": int(len(df)),
                "n_actives": int(y.sum()),
                "feature_dim": int(X.shape[1]),
            }),
            str(self.cfg.resolve("models/classical/model_card.json")),
        )
        self.log.info(f"Exported frozen model artifact -> {out}")
        return out

    def run_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        reps = Representations().build(df)
        sim_matrix = np.load(str(self.cfg.resolve("sar/fingerprints/similarity_morgan.npy")))
        results = {
            "loco": self.loco(reps, df),
            "scaffold_out": self.series_and_scaffold_out(reps, df, "murcko_scaffold"),
            "series_out": self.series_out(reps, df),
            "chemical_distance": self.chemical_distance(reps, df, sim_matrix),
            "bootstrap": self.bootstrap(reps, df),
            "permutation": self.permutation_test(reps, df),
            "applicability_domain": self.applicability_domain(df, sim_matrix),
            "uncertainty_axes": self.uncertainty_axes(reps, df, sim_matrix),
            "frozen_model": self.export_frozen_model(reps, df),
        }
        save_manifest(
            make_provenance("validation", self.cfg, {"n_compounds": int(len(df))}),
            str(self.cfg.resolve("validation/validation_manifest.json")),
        )
        return results


def run_phase4(cfg_path: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    import os
    import logging
    from pathlib import Path
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase4_models", str(cfg.resolve("logs")))
    if df is None:
        p = str(cfg.resolve("data/processed/data_analysis_ready.csv"))
        df = pd.read_csv(p)
        df["mol"] = df["isomeric_SMILES"].apply(Chem.MolFromSmiles)
    val = Validator(cfg, logger)
    return val.run_all(df)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase4(cfgp)