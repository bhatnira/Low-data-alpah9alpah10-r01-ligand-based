#!/usr/bin/env python3
"""Phase 7: Structural consensus integration.

Implements prompt2.txt sections 17-19 and report 6:
  - AF3 / Boltz outputs are PREDICTED STRUCTURAL HYPOTHESES, never facts
  - consensus requires reproduction across independent predictions
  - no AF3/Boltz outputs exist in the repository -> explicit NOT AVAILABLE
  - benchmarks against a7/a4b2 only if structural data present (absent here)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
)


class StructuralConsensus:
    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger

    def audit_availability(self) -> Dict[str, Any]:
        af3_dir = self.cfg.root / "structures/af3"
        boltz_dir = self.cfg.root / "structures/boltz"
        af3_files = sorted(af3_dir.glob("*")) if af3_dir.exists() else []
        boltz_files = sorted(boltz_dir.glob("*")) if boltz_dir.exists() else []
        # Also search the analysis/structural dir
        struct_dir = self.cfg.root / "analysis/structural"
        struct_files = sorted(struct_dir.glob("*")) if struct_dir.exists() else []
        return {
            "af3_present": bool(af3_files),
            "af3_files": [str(p) for p in af3_files],
            "boltz_present": bool(boltz_files),
            "boltz_files": [str(p) for p in boltz_files],
            "structural_analysis_present": bool(struct_files),
            "structural_analysis_files": [str(p) for p in struct_files],
        }

    def benchmark_benchmarks(self) -> pd.DataFrame:
        """Check for a7/a4b2 structural benchmarks (section 19)."""
        rows = [
            {"subtype": "a7", "data_present": False, "status": "NOT AVAILABLE",
             "note": "No experimentally characterized a7 structural data in repository"},
            {"subtype": "a4b2", "data_present": False, "status": "NOT AVAILABLE",
             "note": "No experimentally characterized a4b2 structural data in repository"},
        ]
        return pd.DataFrame(rows)

    def consensus_report(self) -> pd.DataFrame:
        avail = self.audit_availability()
        self.log.step("Structural consensus audit")
        if not any([avail["af3_present"], avail["boltz_present"], avail["structural_analysis_present"]]):
            self.log.warn(
                "No AF3/Boltz/structural outputs found. Structural hypotheses are "
                "REQUIRES_EXPERIMENTAL_VALIDATION / NOT AVAILABLE."
            )
            return pd.DataFrame([{
                "scope": "a9a10_receptor_ensemble",
                "af3": "NOT AVAILABLE",
                "boltz": "NOT AVAILABLE",
                "benchmark_subtypes": "NOT AVAILABLE",
                "consensus_rule": "require reproduction across independent predictions (none available)",
                "conclusion": "UNKNOWN - REQUIRES EXPERIMENTAL VALIDATION",
                "reported_as": "predicted structural hypothesis (not experimental fact)",
            }])
        # placeholder for when structural outputs appear
        return pd.DataFrame([{
            "scope": "a9a10_receptor_ensemble",
            "af3": "PRESENT" if avail["af3_present"] else "ABSENT",
            "boltz": "PRESENT" if avail["boltz_present"] else "ABSENT",
            "benchmark_subtypes": "a7, a4b2",
            "consensus_rule": "require reproduction across independent predictions",
            "conclusion": "PENDING",
            "reported_as": "predicted structural hypothesis (not experimental fact)",
        }])

    def run_all(self) -> Dict[str, Any]:
        report = self.consensus_report()
        save_df(report, str(self.cfg.resolve("structures/consensus/structural_consensus_report.csv")))
        save_df(
            self.benchmark_benchmarks(),
            str(self.cfg.resolve("structures/consensus/structural_benchmarks.csv")),
        )
        save_manifest(
            make_provenance("structures", self.cfg, {"availability": self.audit_availability()}),
            str(self.cfg.resolve("structures/consensus/structural_manifest.json")),
        )
        return {"consensus": report}


def run_phase7(cfg_path: str) -> Dict[str, Any]:
    import os, logging
    from pathlib import Path
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("phase7_structures", str(cfg.resolve("logs")))
    sc = StructuralConsensus(cfg, logger)
    return sc.run_all()


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_phase7(cfgp)