#!/usr/bin/env python3
"""End-to-end workflow orchestrator.

Runs the pipeline phases in dependency order:
    dataset -> sar -> models -> xai -> taf -> structures -> medchem
    -> reinvent -> chemspace -> candidates -> library -> lock -> prospective
    -> loop -> audit -> reports

Usage:
    python3 workflows/run_all.py                # everything
    python3 workflows/run_all.py --only models xai taf
    python3 workflows/run_all.py --config PATH
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_manifest,
)

PHASES: List[Dict[str, Callable]] = [
    {"key": "audit", "fn": lambda cfg: __import__("src.project_audit", fromlist=["run_phase1"]).run_phase1(cfg),
     "desc": "Phase 1: initial project audit (section 3 - inspect before modifying)"},
    {"key": "dataset", "fn": lambda cfg: __import__("src.dataset", fromlist=["run_phase2"]).run_phase2(cfg),
     "desc": "Phase 2: dataset governance + stereochemistry audit"},
    {"key": "sar", "fn": lambda cfg: __import__("src.sar", fromlist=["run_phase3"]).run_phase3(cfg),
     "desc": "Phase 3: experimental SAR (MMP, fragments, scaffolds, hypotheses)"},
    {"key": "sar_ascorbate", "fn": lambda cfg: __import__("src.sar_ascorbate", fromlist=["run_phase_sar_ascorbate"]).run_phase_sar_ascorbate(cfg),
     "desc": "Phase 3b: ascorbate-focused SAR enrichment (R01 aim 1 deliverable)"},
    {"key": "models", "fn": lambda cfg: __import__("src.models", fromlist=["run_phase4"]).run_phase4(cfg),
     "desc": "Phase 4: model panel + LOCO/scaffold-out/bootstrap/permutation"},
    {"key": "selectivity", "fn": lambda cfg: __import__("src.other_modulators", fromlist=["run_phase_selectivity"]).run_phase_selectivity(cfg),
     "desc": "Phase 4b: non-a9a10 nAChR modulator selectivity/decoys panel (curation + chemotype scan + benchmark)"},
    {"key": "xai", "fn": lambda cfg: __import__("src.xai", fromlist=["run_phase5"]).run_phase5(cfg),
     "desc": "Phase 5: explainable AI + stability"},
    {"key": "taf", "fn": lambda cfg: __import__("src.taf", fromlist=["run_phase6"]).run_phase6(cfg),
     "desc": "Phase 6: TAF evidence matrix + falsification framework"},
    {"key": "structures", "fn": lambda cfg: __import__("src.structures", fromlist=["run_phase7"]).run_phase7(cfg),
     "desc": "Phase 7: structural hypotheses (NOT AVAILABLE without AF3/Boltz)"},
    {"key": "medchem", "fn": lambda cfg: __import__("src.medchem", fromlist=["run_phase8"]).run_phase8(cfg),
     "desc": "Phase 8: traditional medicinal chemistry baseline"},
    {"key": "reinvent", "fn": lambda cfg: __import__("src.reinvent", fromlist=["run_phase9_11"]).run_phase9_11(cfg),
     "desc": "Phases 9-11: REINVENT configs + generation plan (no binary -> no molecules)"},
    {"key": "chemspace", "fn": lambda cfg: __import__("src.chemspace", fromlist=["run_phase12"]).run_phase12(cfg),
     "desc": "Phase 12: chemical-space expansion report"},
    {"key": "candidates", "fn": lambda cfg: __import__("src.candidates", fromlist=["run_phase13"]).run_phase13(cfg),
     "desc": "Phase 13: candidate prioritization + portfolio A-F"},
    {"key": "library", "fn": lambda cfg: __import__("src.library", fromlist=["run_phase_library"]).run_phase_library(cfg),
     "desc": "Phase 19: chemotype-library projection + coverage targets (R01 deliverable)"},
    {"key": "lock", "fn": lambda cfg: __import__("src.lock", fromlist=["run_phase14"]).run_phase14(cfg),
     "desc": "Phase 14: model lock before prospective testing"},
    {"key": "prospective", "fn": lambda cfg: __import__("src.prospective", fromlist=["run_phase15_18"]).run_phase15_18(cfg),
     "desc": "Phases 15-18: prospective plan + result ingestion template"},
    {"key": "synth_route", "fn": lambda cfg: __import__("src.synth_route", fromlist=["run_phase_synth_route"]).run_phase_synth_route(cfg),
     "desc": "Phase 4c: retrosynthetic route prediction for high-priority candidates (AIZynthFinder; auto-skips when models absent)"},
    {"key": "loop", "fn": lambda cfg: __import__("src.loop", fromlist=["run_phase_loop"]).run_phase_loop(cfg),
     "desc": "Phases 19-24: closed-loop experimentation (plate design, oversight gate, adaptable updates)"},
    {"key": "audit", "fn": lambda cfg: __import__("src.audit", fromlist=["run_audit"]).run_audit(cfg),
     "desc": "Audit: leakage + 18-question final audit"},
    {"key": "reports", "fn": lambda cfg: __import__("src.reports", fromlist=["run_reports"]).run_reports(cfg),
     "desc": "Reports: build combined HTML report"},
]

ORDER = [p["key"] for p in PHASES]


def main() -> int:
    ap = argparse.ArgumentParser(description="LBM TAF workflow orchestrator")
    ap.add_argument("--config", default=os.environ.get("TAF_CONFIG",
                     str(Path(__file__).resolve().parent.parent / "configs/config.yaml")))
    ap.add_argument("--only", nargs="*", default=None, help="subset of phases to run")
    args = ap.parse_args()

    cfg = ProjectConfig(args.config)
    logger = ManagedLogger("workflow", str(cfg.resolve("logs")))

    only = set(args.only) if args.only else None
    if only:
        bad = only - set(ORDER)
        if bad:
            logger.error(f"unknown phases: {sorted(bad)}; available: {ORDER}")
            return 2
        sel = [p for p in PHASES if p["key"] in only]
    else:
        sel = PHASES

    results: Dict[str, str] = {}
    for phase in sel:
        key = phase["key"]
        logger.step(f"{key} -> {phase['desc']}")
        try:
            phase["fn"](args.config)
            results[key] = "OK"
            logger.info(f"{key} completed")
        except Exception as e:
            logger.error(f"{key} FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[key] = f"FAILED: {e}"
            if not only:
                logger.error("pipeline halted at failing phase (use --only to resume later phases)")
                break

    save_manifest(
        make_provenance("workflow", cfg, {"phases_requested": only or ORDER, "results": results}),
        str(cfg.resolve("reports/workflow_run_manifest.json")),
    )
    logger.info(f"workflow results: {results}")
    return 0 if "FAILED" not in " ".join(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())