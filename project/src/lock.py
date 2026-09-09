#!/usr/bin/env python3
"""Phase 14: Model lock manifest.

Implements prompt2.txt section 28 and report 10:
  - freeze model architecture, weights refs, preprocessing, TAF definitions,
    thresholds, uncertainty/ad criteria, selection algorithm, endpoints,
    statistical analysis, candidate list BEFORE prospective synthesis
  - record git commit, config hash, seeds, model hashes, REINVENT config,
    TAF version, candidate-library hash
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.common import (
    ProjectConfig, ManagedLogger, git_commit, make_provenance, save_manifest, sha256_text,
)


class ModelLock:
    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger

    def _hash_file(self, p: Path) -> Optional[str]:
        if not p.exists():
            return None
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def freeze(self, df: Optional[pd.DataFrame] = None,
               candidate_path: Optional[str] = None) -> Dict[str, Any]:
        self.log.step("Model lock")
        if df is None:
            p = self.cfg.resolve("data/processed/data_analysis_ready.csv")
            df = pd.read_csv(p) if p.exists() else pd.DataFrame()
        if candidate_path is None:
            candidate_path = str(self.cfg.resolve("prospective/prospective_candidates.csv"))

        inputs = {
            "canonical_dataset": self._hash_file(self.cfg.resolve("data/processed/data_analysis_ready.csv")),
            "sar_mmp": self._hash_file(self.cfg.resolve("sar/mmp/mmp_analysis.csv")),
            "taf_evidence": self._hash_file(self.cfg.resolve("taf/evidence/taf_evidence_matrix.csv")),
            "xai_stability": self._hash_file(self.cfg.resolve("xai/stability/xai_stability.csv")),
            "validation_summary": self._hash_file(self.cfg.resolve("validation/loco/loco_summary.csv")),
            "candidate_library": self._hash_file(Path(candidate_path)),
        }

        manifest = {
            "locked_at": make_provenance("model_lock", self.cfg, {})["generated_at"],
            "lock_reason": "Pre-prospective synthesis freeze. NO computational redesign after "
                           "experimental outcomes are revealed except as an explicitly labeled "
                           "exploratory cycle.",
            "git_commit": git_commit(),
            "configuration_hash": make_provenance("model_lock", self.cfg, {})["config_hash"],
            "seeds": self.cfg.seeds,
            "model_architecture": "classical interpretable panel (logreg/ridge/RF/ExtraTrees/GBM)",
            "model_weights_referenced_by": "reproducible retraining from frozen config + data hash",
            "preprocessing": "StandardScaler per fold; fingerprints Morgan r2 2048 + 10 descriptors",
            "taf_version": "v0-prospective",
            "taf_definitions": self.cfg.resolve("taf/evidence/taf_evidence_matrix.csv").as_posix(),
            "thresholds": {
                "active_threshold": "potency>0 AND potentiation>0",
                "applicability_domain": "NEAR/IN by min-NN Tanimoto distance",
                "uncertainty_axis_separated": True,
            },
            "candidate_selection_algorithm": "portfolio groups A-F; multi-axis; NO single-score sorting",
            "primary_endpoint": str(self.cfg.raw["prospective"]["primary_endpoint"]),
            "statistical_analysis_plan": ("predefined responder criteria; effect-size RD with CI; Wilson/exact "
                              "binomial CIs; 10k bootstrap; Benjamini-Hochberg FDR; sensitivity analyses; "
                              "full plan frozen in prospective/roundN_lock.json (section 31)"),
            "candidate_library_hash": inputs["candidate_library"],
            "input_hashes": inputs,
            "reinvent_config_hashes": {
                str(p.relative_to(self.cfg.root)): self._hash_file(p)
                for p in (self.cfg.root / "reinvent").rglob("*")
                if p.is_file() and "out" not in p.parts
            } if (self.cfg.root / "reinvent").exists() else {},
            "prior_file_hashes": {
                str(p.relative_to(self.cfg.root)): self._hash_file(p)
                for p in (self.cfg.root / "priors").glob("*.prior")
            } if (self.cfg.root / "priors").exists() else {},
        }
        save_manifest(manifest, str(self.cfg.resolve("prospective/model_lock_manifest.json")))
        save_manifest(manifest, str(self.cfg.resolve("reports/model_lock_manifest.json")))
        self.log.info("Model lock manifest written; prospective cycle is FROZEN.")
        return manifest

    def verify(self) -> Dict[str, Any]:
        manifest_path = self.cfg.resolve("prospective/model_lock_manifest.json")
        if not manifest_path.exists():
            return {"locked": False, "reason": "no manifest found"}
        with open(manifest_path) as f:
            m = json.load(f)
        return {"locked": True, "manifest": m}


def run_phase14(cfg_path: str, df: Optional[pd.DataFrame] = None,
                candidate_path: Optional[str] = None) -> Dict[str, Any]:
    import logging
    from pathlib import Path
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase14_lock", str(cfg.resolve("logs")))
    lock = ModelLock(cfg, logger)
    return lock.freeze(df=df, candidate_path=candidate_path)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase14(cfgp)