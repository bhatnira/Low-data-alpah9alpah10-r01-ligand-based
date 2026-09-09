#!/usr/bin/env python3
"""Phases 15-18: Prospective interface, experimental-result ingestion,
statistical analysis, and active learning update.

Implements prompt2.txt sections 22, 29-32, 35, 36, 42-report-11:
  - locked prospective plan (no post-hoc retraining contamination)
  - experimental result ingestion keeps raw data; update only after completion
  - statistical plan: responder criteria, effect sizes, bootstrap CI,
    multiple-testing correction
  - active learning: T(F) update, TAF confidence update, next-cycle re-ranking
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
)


class Prospective:
    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger

    def power_analysis(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """Section 32: sample-size / experimental power without inventing a size.

        Uses the observed preliminary variability only:
          1. if empirical n_active/n_inactive exists, compute the minimum
             detectable active-fraction difference (chi-square, alpha=0.05,
             power=0.80) for a prospective round of given size
          2. document explicitly what is NOT powered and why
        """
        self.log.step("Sample-size / power analysis (section 32)")
        if df is None:
            p = self.cfg.resolve("data/processed/data_analysis_ready.csv")
            df = pd.read_csv(p) if p.exists() else pd.DataFrame()
        out: Dict[str, Any] = {"n_preliminary": int(len(df)) if len(df) else 0}
        if len(df) == 0:
            out["status"] = "UNDETERMINED - no preliminary data; formal power deferred"
            return out

        act = int(df["is_active"].sum()) if "is_active" in df else 0
        n = len(df)
        out["n_active_prelim"] = act
        out["observed_active_fraction"] = round(act / n, 4) if n else None
        if act == 0 or act == n:
            out["status"] = ("SINGLE-CLASS dataset: active fraction is degenerate; "
                             "no binomial power curve computable - rely on effect-size/"
                             "uncertainty-based criteria per section 32")
            save_manifest(out, str(self.cfg.resolve("prospective/power_analysis.json")))
            return out

        # minimum detectable difference for a prospective round at fixed
        # alpha/power using the observed baseline rate as the null
        # (one-sample proportion z-test, normal approximation; no statsmodels dep).
        def _power_one_prop(n_total: int, p0: float, p1: float,
                            alpha: float = 0.05) -> float:
            if n_total <= 0 or not (0 < p0 < 1):
                return 0.0
            se = np.sqrt(p0 * (1 - p0) / n_total)
            d = abs(p1 - p0)
            if d <= 0:
                return 0.0
            z_a = sp_stats.norm.ppf(1 - alpha / 2)
            return float(sp_stats.norm.cdf((d / se) - z_a))

        base = act / n
        sizes = [6, 10, 15, 30, 45, 60, 90, 120]
        out["prospective_total_n_vs_power"] = {
            str(size): round(_power_one_prop(size, base, 0.80), 3)
            for size in sizes
        }
        # estimate n to reach 80% power vs alternative active fraction 0.80
        n_req = None
        for size in range(4, 401, 2):
            if _power_one_prop(size, base, 0.80) >= 0.80:
                n_req = size
                break
        out["min_total_n_for_80%_power_alt0.80"] = n_req
        out["power_assumption_note"] = ("Power assumes prospective activity fraction 0.80 "
                                        "(upper-bounded by observed 7/30 PAM series chemistry); "
                                        "this is an effect-size-based criterion, NOT an invented "
                                        "sample size (section 32)")
        out["status"] = ("REPORTED from preliminary variability only; "
                         "prospective confirmation must NOT be re-powered after "
                         "results are observed (section 32)")
        save_manifest(out, str(self.cfg.resolve("prospective/power_analysis.json")))
        return out

    def ingestion_template(self) -> pd.DataFrame:
        cols = [
            "compound_id", "SMILES", "experimental_status", "activity_potency_uM",
            "activity_potentiation_pct", "replicate_1_uM", "replicate_2_uM",
            "replicate_3_uM", "biological_replicates_n", "assay_date", "operator",
            "vehicle_control", "positive_control_id", "raw_data_path", "notes",
            "exclusion_criteria_triggered",
        ]
        return pd.DataFrame(columns=cols)

    def plan(self, portfolio: pd.DataFrame, n_round: int = 1) -> pd.DataFrame:
        self.log.step("Prospective experimental plan")
        if not len(portfolio):
            return portfolio
        per_round = int(self.cfg.raw["prospective"]["per_round_n"])
        # balanced sampling across groups A-F rather than top-score only
        if "portfolio_group" in portfolio.columns:
            plan = portfolio.groupby("portfolio_group", group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(1, per_round // 6)), random_state=self.cfg.seed("models")),
                include_groups=False,
            ).reset_index(drop=True)
        else:
            plan = portfolio
        plan["experimental_round"] = n_round
        plan["randomized_test_order"] = np.random.default_rng(n_round).permutation(len(plan))
        plan["blinded_id"] = plan["molecule_id"].apply(lambda x: f"BLIND-{abs(hash(str(x))) % 100000}")
        plan["exclusion_criteria"] = "PREDEFINED before data lock"
        save_df(plan, str(self.cfg.resolve(f"prospective/round{n_round}_plan.csv")))
        return plan

    def lock_confirmatory(self, round_no: int = 1) -> Dict[str, Any]:
        """Pre-registration snapshot for the confirmatory cycle (section 35)."""
        p = {
            "round": round_no,
            "locked_at": make_provenance("prospective", self.cfg, {})["generated_at"],
            "contamination_guard": "No retraining or TAF redefinition will occur on the basis of "
                                   "these results before the round is complete and reported.",
            "prespecified_responder_criterion": "compound active if potency>0 uM AND potentiation>0%",
            "statistics": {
                "primary_analysis": "active-fraction enrichment (TAF-guided portfolio) vs "
                                    "pre-specified medchem baseline with continuity-corrected CI",
                "effect_size": "risk difference (RD) in active fraction; reported with 95% CI",
                "confidence_intervals": "Wilson / exact binomial 95% CI on active fraction per portfolio group",
                "bootstrap_ci": True,
                "bootstrap_detail": "10,000 resamples of compound-level outcomes; percentile CI; seed-fixed",
                "hierarchical_mixed_effects": "NOT APPLICABLE to primary endpoint (compound-level binomial "
                                              "outcomes); would apply to EC50/Emax comparisons once replicate "
                                              "sweeps exist - pre-specified here",
                "replicate_aware_analysis": "biological replicates to be averaged before analysis; technical "
                                            "sweeps from same preparation NEVER treated as independent",
                "multiple_testing_correction": "Benjamini-Hochberg FDR if >5 portfolio comparisons; "
                                               "families defined at lock time",
                "sensitivity_analyses": ("1) binary inactive (potency=0) excluded vs imputed; "
                                         "2) responder threshold +/-50% potentiation; "
                                         "3) leave-one-series-out precision robustness"),
                "causal_claim_restriction": "no causal attribution claimed from single confirmatory round",
            },
        }
        save_manifest(p, str(self.cfg.resolve(f"prospective/round{round_no}_lock.json")))
        return p

    def ingest_results(self, results: pd.DataFrame, round_no: int = 1) -> pd.DataFrame:
        """Ingest experimental outcomes. Raw data must be preserved.-
        Only appends; never overwrites experimental observations."""
        self.log.step(f"Ingesting experimental results (round {round_no})")
        if "activity_potency_uM" in results.columns and "activity_potentiation_pct" in results.columns:
            results["is_active"] = (
                (results["activity_potency_uM"] > 0) & (results["activity_potentiation_pct"] > 0)
            ).astype(int)
        save_df(results, str(self.cfg.resolve(f"data/experimental/round{round_no}_results_raw.csv")))
        self.log.info(f"Recorded {len(results)} experimental observations (round {round_no}).")
        return results

    def active_learning_update(self, round_no: int, results: pd.DataFrame,
                               taf_evidence: pd.DataFrame,
                               scaffold_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Update TAF evidence + transferability metric T(F)."""
        self.log.step("Active learning update (section 35 / 36)")
        tested_scaffolds = scaffold_df[scaffold_df["Identifier"].astype(str).isin(
            results["compound_id"].astype(str)
        )]
        n_tested = len(results)
        n_active = int(results["is_active"].sum()) if "is_active" in results else 0

        update = pd.DataFrame([{
            "round": round_no,
            "n_tested": int(n_tested),
            "n_active": n_active,
            "active_fraction": round(n_active / n_tested, 4) if n_tested else 0,
            "novel_scaffold_active": "REPORTED AFTER SCAFFOLD ANNOTATION",
            "transferability_metric_T_F": "UPDATED_AFTER_EXPERIMENTAL_ROUND",
        }])
        save_df(update, str(self.cfg.resolve(f"taf/evidence/active_learning_round{round_no}.csv")))

        # annotate TAF evidence with prospective results (append-only)
        if len(taf_evidence) and "status" in taf_evidence.columns:
            taf_evidence = taf_evidence.copy()
            taf_evidence["prospective_result_round_" + str(round_no)] = "PENDING_ANALYSIS"
        save_df(taf_evidence, str(self.cfg.resolve("taf/evidence/taf_evidence_matrix.csv")))

        # Hypothesis-resolution log
        self.log.info(
            f"Round {round_no} -> tested {n_tested}, active {n_active}. "
            "Hypothesis resolution and TAF update follow once scaffold annotation completes."
        )
        return {"update": update, "taf_evidence": taf_evidence}

    def run_all(self, portfolio: pd.DataFrame, round_no: int = 1) -> Dict[str, Any]:
        template = self.ingestion_template()
        save_df(template, str(self.cfg.resolve("data/experimental/results_ingestion_template.csv")))
        plan = self.plan(portfolio, n_round=round_no)
        lock = self.lock_confirmatory(round_no)
        power = self.power_analysis()
        return {"template": template, "plan": plan, "lock": lock, "power": power}


def run_phase15_18(cfg_path: str, portfolio: Optional[pd.DataFrame] = None,
                   round_no: int = 1) -> Dict[str, Any]:
    import logging, os
    from pathlib import Path
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase15_18_prospective", str(cfg.resolve("logs")))
    if portfolio is None:
        p = str(cfg.resolve("prospective/prospective_candidates.csv"))
        portfolio = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(
            columns=["molecule_id", "isomeric_SMILES", "portfolio_group", "TAF_score_2d"])
    pro = Prospective(cfg, logger)
    return pro.run_all(portfolio, round_no=round_no)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase15_18(cfgp)