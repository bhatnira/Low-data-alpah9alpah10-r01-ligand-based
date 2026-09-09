#!/usr/bin/env python3
"""HTML report generation for the 11 required reports (section 42).

- Report 1  - Dataset QC
- Report 2  - Experimental SAR
- Report 3  - Model benchmarking
- Report 4  - XAI reproducibility
- Report 5  - TAF evidence
- Report 6  - Structural hypotheses
- Report 7  - Generative chemistry
- Report 8  - Chemical-space expansion
- Report 9  - Candidate prioritization
- Report 10 - Model lock
- Report 11 - Prospective validation
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common import ProjectConfig, ManagedLogger


def _render_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df is None or len(df) == 0:
        return "<p><em>No data available.</em></p>"
    d = df.head(max_rows)
    cols = [c for c in d.columns if not str(c).startswith("Unnamed")]
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = ""
    for _, r in d.iterrows():
        body += "<tr>" + "".join(
            f"<td>{html.escape(str(r[c]))[:120]}</td>" for c in cols
        ) + "</tr>"
    return f"<table border='1' cellpadding='4'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


class ReportBuilder:
    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger
        self.reports: List[str] = []

    def _resolve(self, rel: str) -> Path:
        p = self.cfg.root / rel
        return p

    def _read(self, rel: str) -> pd.DataFrame:
        p = self._resolve(rel)
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _read_json(self, rel: str) -> pd.DataFrame:
            p = self._resolve(rel)
            if not p.exists():
                return pd.DataFrame()
            try:
                import json as _json
                with open(p) as fh:
                    return pd.DataFrame([_json.load(fh)])
            except Exception:
                return pd.DataFrame()

    def build_all(self) -> str:
        sections = []
        sections.append(("Report 1 - Dataset QC", self._read("data/experimental/canonical_experimental_dataset.csv"),
                         "Canonical experimental dataset (governance fields + stereochemistry)."))
        sections.append(("Report 2 - Experimental SAR", self._read("sar/mmp/mmp_transformations.csv"),
                        "Matched-molecular-pair transformations with activity changes."))
        sections.append(("Report 3 - Model benchmarking", self._read("validation/loco/loco_summary.csv"),
                        "Leave-one-compound-out benchmarking across the model panel."))
        sections.append(("Report 4 - XAI reproducibility", self._read("xai/stability/xai_stability.csv"),
                        "Stability of explanations across models x seeds x splits x methods."))
        sections.append(("Report 4b - XAI explanation preservation", self._read("xai/explanation_preservation.csv"),
                        "Top-feature composition vs generic descriptor dominance (artifact check)."))
        sections.append(("Report 4c - Counterfactual probes", self._read("xai/counterfactual/counterfactual.csv"),
                        "15 hypothesis untested probes across 7 perturbation categories."))
        sections.append(("Report 11b - Prospective power analysis", self._read_json("prospective/power_analysis.json"),
                        "Pre-specified power/sample-size analysis (no post-hoc re-powering)."))
        sections.append(("Report 5 - TAF evidence", self._read("taf/evidence/taf_evidence_matrix.csv"),
                        "TAF evidence matrix with evidence classes and falsification criteria."))
        sections.append(("Report 6 - Structural hypotheses", self._read("structures/consensus/structural_consensus_report.csv"),
                        "Structural consensus. No AF3/Boltz outputs present -> NOT AVAILABLE."))
        sections.append(("Report 7 - Generative chemistry", self._read("reinvent/generation_plan.csv"),
                        "REINVENT generation plan + config skeletons."))
        sections.append(("Report 8 - Chemical-space expansion", self._read("chemical_space/chemical_space_report.csv"),
                        "Chemical-space expansion metrics."))
        sections.append(("Report 9 - Candidate prioritization", self._read("prospective/prospective_candidates.csv"),
                        "Prospective candidate portfolio (groups A-F)."))
        sections.append(("Report 10 - Model lock", self._read("prospective/model_lock_manifest_from_json"), ""))
        sections.append(("Report 11 - Prospective validation", self._read("data/experimental/results_ingestion_template.csv"),
                        "Template for experimental result ingestion; actual results pending rounds."))

        part_rows = []
        for name, df, desc in sections:
            if name.startswith("Report 10") and len(df) == 0:
                df = self._manifest_as_df("prospective/model_lock_manifest.json")
                desc = "Model lock manifest - frozen before prospective synthesis."
            part = f"<h2>{html.escape(name)}</h2><p>{html.escape(desc)}</p>{_render_table(df)}"
            parts_row = f"<div class='report'>{part}</div>"
            part_rows.append(parts_row)

        # Leakage + final audit as appendix
        appendix = []
        appendix.append(("Appendix A - Leakage audit", self._read("reports/leakage_audit.csv"),
                         "Explicit leakage checks (section 40)."))
        appendix.append(("Appendix B - Final scientific audit", self._read("reports/final_scientific_audit.csv"),
                        "Answers to the 18 'before you finish' questions (section 51)."))
        appendix.append(("Appendix C - Failure analysis", self._read("reports/failure_analysis.csv"),
                        "Ten explicit failure modes with tests, determinations, and alternative strategies (section 34)."))
        appendix.append(("Appendix D - Strategy comparison", self._read("reports/strategy_comparison.csv"),
                        "Comparison of medicinal-chemistry strategies (section 46)."))
        appendix.append(("Appendix E - Overclaim detection", self._read("reports/overclaim_detection.csv"),
                        "Automated scan for overclaiming language across all generated artifacts (section 38)."))
        appendix.append(("Appendix F - Medicinal-chemistry oversight checklist",
                        self._read("reports/medchem_oversight_checklist.csv"),
                        "Expert med-chem review checklist; all items PENDING_REVIEW until a human signs off (section 37)."))
        for name, df, desc in appendix:
            part_rows.append(f"<div class='report'><h2>{html.escape(name)}</h2><p>{html.escape(desc)}</p>{_render_table(df)}</div>")

        # Appendix G - closed-loop experimentation artifacts (section 35 / loop phase)
        loop_sections = [
            ("G1 - Explanation consensus (Borda)", "loop/xai/explanation_consensus.csv",
             "Borda fusion of RF-Gini / permutation / ablation importances with pairwise method agreement."),
            ("G2 - Stereochemical probes (TAF-4)", "loop/stereo/stereo_probes_round0.csv",
             "Experimental enantiomer/diastereomer probes designed from active parents to make TAF-4 testable."),
            ("G3 - Round plate 1", "loop/round1_plate.csv",
             "Fixed-size balanced plate (portfolio groups A-F + stereo probes); blinded IDs; PENDING_REVIEW."),
            ("G4 - Round 1 oversight gate", "loop/round1_oversight_gate.csv",
             "Mandatory med-chem sign-off gate; experiment BLOCKED until APPROVED."),
            ("G5 - Generator + structural data watch", "loop/data_watch.json",
             "Honest availability scan; REINVENT/AF3/Boltz ingest when outputs exist, else NOT AVAILABLE."),
            ("G6 - Cycle ledger", "loop/cycle_ledger.json",
             "Append-only provenance of every closed-loop cycle (design, gate, pooling, refit, watch)."),
        ]
        for name, rel, desc in loop_sections:
            loop_df = self._read(rel) if rel.endswith(".csv") else self._read_json(rel)
            if len(loop_df):
                part_rows.append(f"<div class='report'><h2>{html.escape(name)}</h2>"
                                 f"<p>{html.escape(desc)}</p>{_render_table(loop_df)}</div>")

        # Appendix H - chemotype library (R01 deliverable, phase 19)
        lib_sections = [
            ("H1 - Chemotype families", "loop/library/chemotype_families.csv",
             "Chemotype families = distinct Bemis-Murcko scaffolds; tiers (1 novel+clean, 2 hypothesis, 3 residual)."),
            ("H2 - Coverage targets", "loop/library/coverage_targets.csv",
             "Pre-registered round-by-round library coverage targets and go/no-go criteria."),
            ("H3 - External validation status", "loop/library/external_validation_status.json",
             "Honest status of models validated on published/external datasets."),
            ("H4 - Pre-registration round 1", "loop/pre_registration_round1.json",
             "Analysis criteria frozen BEFORE experiment (comparator arms, EF threshold, FDR, go/no-go)."),
        ]
        for name, rel, desc in lib_sections:
            lib_df = self._read(rel) if rel.endswith(".csv") else self._read_json(rel)
            if len(lib_df):
                part_rows.append(f"<div class='report'><h2>{html.escape(name)}</h2>"
                                 f"<p>{html.escape(desc)}</p>{_render_table(lib_df)}</div>")

        # Appendix I - ascorbate-focused SAR enrichment (R01 aim 1, phase 3b)
        asc_sections = [
            ("I1 - Ascorbate core presence", "sar/ascorbate/ascorbate_core_presence.csv",
             "Per-compound ascorbate-core detection (SMARTS OC1=C(O)C(=O)OC1-[#6]), 2-O/3-O enol substitution (standard ascorbate numbering), chain modifications, L/D reference-probe roles."),
            ("I2 - Ascorbate sub-series SAR", "sar/ascorbate/ascorbate_subseries.csv",
             "Sub-series strata (free ascorbate reference / 3-O-alkoxy / 2-O-alkoxy / acetonide / 6-bromo / 6-O-benzyl / non-ascorbate) with active fractions and potency ranges. PI notes: 3-O-substituted ascorbate is the high-potency arm (ID12 0.198 uM), 6-bromo-6-deoxy second (ID25 2.63 uM)."),
            ("I3 - Ascorbate-restricted MMP", "sar/ascorbate/ascorbate_mmp.csv",
             "Matched molecular pairs where BOTH arms preserve the ascorbate gamma-lactone enediol core."),
            ("I4 - Ascorbate enrichment manifest", "sar/ascorbate/ascorbate_manifest.json",
             "Provenance of the ascorbate SAR enrichment phase: core SMARTS, sub-series counts, panel linkage."),
        ]
        for name, rel, desc in asc_sections:
            asc_df = self._read(rel) if rel.endswith(".csv") else self._read_json(rel)
            if len(asc_df):
                part_rows.append(f"<div class='report'><h2>{html.escape(name)}</h2>"
                                 f"<p>{html.escape(desc)}</p>{_render_table(asc_df)}</div>")

        html_template = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>alpha9alpha10 nAChR PAM - TAF Workflow Reports</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 24px; }}
h1 {{ color: #1a3c6e; }}
.report {{ margin: 24px 0; padding: 12px; border: 1px solid #ccc; border-radius: 6px; }}
table {{ border-collapse: collapse; font-size: 12px; }}
th {{ background: #eef2f7; }}
</style></head>
<body>
<h1>alpha9alpha10 nAChR PAM &mdash; NIH-R01-oriented computational workflow</h1>
<p>Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}. All predicted/structural/generated
content is labelled PROSPECTIVE or UNKNOWN until experimental validation (prompt2 section 45).</p>
{''.join(part_rows)}
</body></html>"""
        out = self.cfg.resolve("reports/index.html")
        out.write_text(html_template, encoding="utf-8")
        self.log.info(f"Reports written to {out}")
        return html_template

    def _manifest_as_df(self, rel: str) -> pd.DataFrame:
        import json
        p = self._resolve(rel)
        if not p.exists():
            return pd.DataFrame()
        with open(p) as f:
            m = json.load(f)
        flat = {}
        for k, v in m.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
            else:
                flat[k] = v
        return pd.DataFrame([{"key": k, "value": str(v)[:150]} for k, v in flat.items()])


def run_reports(cfg_path: str) -> str:
    import logging
    from src.common import ProjectConfig, ManagedLogger
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("reports", str(cfg.resolve("logs")))
    rb = ReportBuilder(cfg, logger)
    return rb.build_all()


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    cfgp = os.environ.get("TAF_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
    run_reports(cfgp)