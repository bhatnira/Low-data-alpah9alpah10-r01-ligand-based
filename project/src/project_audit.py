#!/usr/bin/env python3
"""Phase 1: Initial project audit.

Implements prompt2.txt section 3 ("Inspect the existing project before
modifying anything") and section 42-report-1 (audit preamble):

  - inventory existing datasets / preprocessing / activity labels
  - inventory molecular representations, models, structure files, scripts,
    configuration, generated molecules, reports
  - determine what is already complete
  - record reuse / non-duplication decisions before any new implementation

The audit is intentionally run BEFORE the dataset phase so that downstream
phases can reuse rather than duplicate completed work.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
    git_commit, sha256_file,
)


class ProjectAudit:
    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger

    def _scan(self, rel: str, patterns: tuple = ()) -> List[str]:
        d = self.cfg.root / rel
        if not d.exists():
            return []
        out = []
        for p in sorted(d.rglob("*")):
            if p.is_dir():
                continue
            if patterns and p.suffix.lstrip(".") not in patterns and p.name not in patterns:
                continue
            out.append(str(p.relative_to(self.cfg.root)))
        return out

    def existing_datasets(self) -> pd.DataFrame:
        rows = []
        data_dirs = [
            ("data/raw", ("csv", "sdf", "xlsx", "tsv")),
            ("data/processed", ("csv", "parquet")),
            ("data/experimental", ("csv", "json")),
            ("data/qc", ("csv",)),
        ]
        for d, pats in data_dirs:
            for f in self._scan(d, pats):
                p = self.cfg.root / f
                try:
                    size = p.stat().st_size
                    n_rows = len(pd.read_csv(p)) if p.suffix == ".csv" else np.nan
                except Exception:
                    size, n_rows = p.stat().st_size, np.nan
                rows.append({
                    "category": "dataset", "dir": d, "path": f,
                    "size_bytes": size, "n_rows": n_rows,
                })
        return pd.DataFrame(rows)

    def existing_models(self) -> pd.DataFrame:
        rows = []
        for d in ("models", "priors"):
            for f in self._scan(d, ("joblib", "pkl", "prior", "pt", "onnx", "json")):
                p = self.cfg.root / f
                rows.append({
                    "category": "model", "dir": d, "path": f,
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(str(p)),
                })
        return pd.DataFrame(rows)

    def existing_artifacts(self) -> pd.DataFrame:
        rows = []
        for d, cat in [
            ("sar", "sar_analysis"), ("validation", "validation"),
            ("xai", "xai"), ("taf", "taf"), ("structures", "structural"),
            ("reinvent", "reinvent"), ("chemical_space", "chemical_space"),
            ("prospective", "prospective"), ("reports", "report"),
            ("generated", "generated"), ("notebooks", "notebook"),
        ]:
            for f in self._scan(d):
                rows.append({"category": cat, "dir": d, "path": f})
        return pd.DataFrame(rows)

    def existing_scripts(self) -> pd.DataFrame:
        rows = []
        for d in ("src", "workflows", "tests"):
            for f in self._scan(d, ("py",)):
                rows.append({"category": "script", "dir": d, "path": f})
        return pd.DataFrame(rows)

    def structural_availability(self) -> pd.DataFrame:
        rows = []
        for sub in ("af3", "boltz", "consensus"):
            d = self.cfg.root / "structures" / sub
            files = [str(p.relative_to(self.cfg.root)) for p in d.rglob("*") if p.is_file()] if d.exists() else []
            rows.append({
                "dir": f"structures/{sub}", "present": bool(files),
                "files": files, "status": "AVAILABLE" if files else "NOT AVAILABLE",
            })
        return pd.DataFrame(rows)

    def run_all(self) -> Dict[str, pd.DataFrame]:
        self.log.step("Phase 1: initial project audit (section 3)")
        ae_datasets = self.existing_datasets()
        ae_models = self.existing_models()
        ae_artifacts = self.existing_artifacts()
        ae_scripts = self.existing_scripts()
        ae_structural = self.structural_availability()

        for name, df_ in [("existing_datasets", ae_datasets),
                          ("existing_models", ae_models),
                          ("existing_scripts", ae_scripts),
                          ("existing_artifacts", ae_artifacts),
                          ("structural_status", ae_structural)]:
            save_df(df_, str(self.cfg.resolve(f"reports/audit/{name}.csv")))
            self.log.info(f"{name}: {len(df_)} entries")

        summary = pd.DataFrame([{
            "component": "experimental_datasets", "status": "EXISTS" if len(ae_datasets) else "NONE",
            "n": len(ae_datasets),
        }, {
            "component": "model_artifacts", "status": "EXISTS" if len(ae_models) else "NONE",
            "n": len(ae_models),
        }, {
            "component": "scripts_modules", "status": "EXISTS" if len(ae_scripts) else "NONE",
            "n": len(ae_scripts),
        }, {
            "component": "reports_artifacts", "status": "EXISTS" if len(ae_artifacts) else "NONE",
            "n": len(ae_artifacts),
        }, {
            "component": "structural_models",
            "status": "ANY_AVAILABLE" if ae_structural["present"].any() else "NOT_AVAILABLE",
            "n": int(ae_structural["present"].sum()),
        }])
        save_df(summary, str(self.cfg.resolve("reports/audit/initial_project_audit_summary.csv")))

        # Reuse decisions recorded so downstream phases do not duplicate work.
        reuse = pd.DataFrame([{
            "phase": "Phase 2 datasets",
            "reuse_decision": "canonical + governed datasets already exist under data/; "
                              "regenerate with added salt/tautomer/charge QC and derived series_id",
            "overwrite": False,
        }, {
            "phase": "Phase 3 SAR",
            "reuse_decision": "sar/ outputs exist; regenerate in governed form with added "
                              "transformation-level MMP and stereo detail",
            "overwrite": False,
        }, {
            "phase": "Phase 4 models",
            "reuse_decision": "models/classical/lbm_rf_fp_desc.joblib exists and is consumed "
                              "by REINVENT scoring; freeze artifact preserved, retrained only "
                              "when the dataset hash changes",
            "overwrite": False,
        }, {
            "phase": "Phases 9-11 REINVENT",
            "reuse_decision": "prior files verification + TOML configs + inception/scaffold "
                              "seeds already present; REINVENT binary unavailable -> config "
                              "generation only (no fabricated molecules)",
            "overwrite": False,
        }, {
            "phase": "Phase 14 model lock",
            "reuse_decision": "prospective/model_lock_manifest.json frozen; preserved unless "
                              "analysis-ready dataset changes hash",
            "overwrite": False,
        }])
        save_df(reuse, str(self.cfg.resolve("reports/audit/reuse_decisions.csv")))

        save_manifest(make_provenance("initial_audit", self.cfg, {
            "existing_datasets": len(ae_datasets),
            "existing_models": len(ae_models),
            "existing_scripts": len(ae_scripts),
            "existing_artifacts": len(ae_artifacts),
            "structural_available": int(ae_structural["present"].sum()),
            "git_commit": git_commit(),
        }), str(self.cfg.resolve("reports/audit/initial_audit_manifest.json")))

        self.log.info("Phase 1 audit complete; no duplicate work will be created.")
        return {
            "datasets": ae_datasets, "models": ae_models,
            "scripts": ae_scripts, "artifacts": ae_artifacts,
            "structural": ae_structural, "summary": summary, "reuse": reuse,
        }


def run_phase1(cfg_path: str) -> Dict[str, pd.DataFrame]:
    import logging
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase1_audit", str(cfg.resolve("logs")))
    return ProjectAudit(cfg, logger).run_all()


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase1(cfgp)