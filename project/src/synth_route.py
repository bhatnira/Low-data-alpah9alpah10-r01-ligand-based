#!/usr/bin/env python3
"""Phase: synth_route - retrosynthetic route prediction for high-priority
alpha9alpha10 PAM candidates (R01 Aim 1/2 support; decision aid for the
oversight gate).

Availability-gated like `structures.py` / `reinvent`:
  - if AIZynthFinder is NOT installed or its policy/template/stock models are
    not present in `synth_route/models`, the phase emits an honest
    NOT_AVAILABLE manifest and returns normally (the pipeline does not halt).
  - if the models ARE present, each target SMILES is given a retrosynthetic
    tree-search; the top ranked multi-step routes, their route probabilities,
    the estimated number of steps, and building-block availability are
    written under <project>/synth_route/.

Honesty contract:
  * Route predictions are PREDICTED_ROUTE / UNVERIFIED - they are computer
    plans from a pretrained model, never a guarantee that a synthesis works.
  * They feed the medicinal-chemistry oversight gate as decision support and
    do NOT replace human approval.
  * No route is fabricated: if AIZynthFinder or the model files are absent,
    no placeholder routes are written.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common import (
    ProjectConfig, ManagedLogger, make_provenance, save_df, save_manifest,
)

MODEL_FILES = {
    "policy": "uspto_model.onnx",
    "templates": "uspto_templates.csv.gz",
    "stock": "zinc_stock.hdf5",
    "ringbreaker": "uspto_ringbreaker_model.onnx",
    "filter": "uspto_filter_model.onnx",
    "config": "config.yml",
}


class SynthRoutePlanner:
    """Retrosynthetic route prediction for a target candidate list.

    AIZynthFinder itself runs in a dedicated Python 3.12 venv
    (project/venvs/aizynth312) because it does not support Python 3.14
    (the interpreter used to run the rest of the pipeline).  Models are the
    public USPTO-onnx + ZINC stock download placed under synth_route/models/
    with an auto-generated config.yml (see docstring - download_public_data).
    """

    def __init__(self, cfg: ProjectConfig, logger: ManagedLogger):
        self.cfg = cfg
        self.log = logger
        self.model_dir = cfg.root / "synth_route/models"
        self.venv_python = cfg.root / "venvs/aizynth312/bin/python"
        self._finder = None  # lazy AIZynthFinder instance

    # ------------------------------------------------------------------
    # availability audit (mirrors structures/reinvent gate)
    # ------------------------------------------------------------------
    def audit_availability(self) -> Dict[str, Any]:
        """Report whether AIZynthFinder + model files are available."""
        import importlib.util
        spec = importlib.util.find_spec("aizynthfinder")
        pkg_present = spec is not None or self.venv_python.exists()
        present = {k: (self.model_dir / v).exists() for k, v in MODEL_FILES.items()}
        models_present = bool(present) and all(present.values())
        return {
            "aizynthfinder_installed": pkg_present,
            "venv_aizynth312": self.venv_python.exists(),
            "model_files": {k: (self.model_dir / v).exists() for k, v in MODEL_FILES.items()},
            "models_present": models_present,
            "available": bool(pkg_present) and models_present,
            "note": ("PREDICTED_ROUTE / UNVERIFIED route plans; decision support "
                     "for the med-chem oversight gate, not a synthesis guarantee"),
        }

    def _init_finder(self) -> Optional[Any]:
        """Lazily instantiate AIZynthFinder from the on-disk models."""
        if self._finder is not None:
            return self._finder
        try:
            from aizynthfinder.aizynthfinder import AiZynthFinder
            finder = AiZynthFinder()
            finder.load(configfile=str(self.model_dir / "config.yml"))
            self._finder = finder
            return finder
        except Exception as e:  # pragma: no cover - depends on external install
            self.log.warn(f"AIZynthFinder init failed: {e}")
            return None

    @staticmethod
    def _resolved_root(cfg) -> str:
        # fallback to a path that does not hard-code the mac username
        return str(cfg.root)

    # ------------------------------------------------------------------
    # target collection
    # ------------------------------------------------------------------
    def collect_targets(self, round_no: int = 1) -> pd.DataFrame:
        """Gather candidate SMILES to plan routes for.

        Precedence: round plan -> round plate -> annotated library top tier.
        Returns a DataFrame with `molecule_id` and `isomeric_SMILES`.
        """
        candidates = [None] * 4
        candidates[0] = ("prospective/round%d_plan.csv" % round_no, ["molecule_id", "isomeric_SMILES"])
        candidates[1] = ("prospective/round%d_lock.json" % round_no, ["molecule_id", "isomeric_SMILES"])
        candidates[2] = ("loop/round%d_plate.csv" % round_no, ["molecule_id", "isomeric_SMILES"])
        candidates[3] = ("loop/library/chemotype_library_annotated.csv",
                         ["molecule_id", "isomeric_SMILES"])

        for rel, cols in candidates:
            p = self.cfg.root / rel
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p) if p.suffix == ".csv" else _load_json_table(p)
            except Exception as e:
                self.log.warn(f"could not read {rel}: {e}")
                continue
            if not {"molecule_id", "isomeric_SMILES"}.issubset(df.columns):
                # try to synthesize the columns from available names
                rename = self._prune_dtypes(df)
                if not {"molecule_id", "isomeric_SMILES"}.issubset(rename.columns):
                    continue
                df = rename
            df = df[["molecule_id", "isomeric_SMILES"]].dropna()
            df = df.drop_duplicates(subset="isomeric_SMILES")
            self.log.info(f"targets from {rel}: {len(df)}")
            return df.reset_index(drop=True)
        self.log.warn("no candidate file found; synth_route has no targets")
        return pd.DataFrame(columns=["molecule_id", "isomeric_SMILES"])

    @staticmethod
    def _prune_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        for c in df.columns:
            low = c.lower()
            if "molecule" in low or "smil" in low or "structure" in low:
                rename[c] = "smiles_source"
        if "molecule_id" not in df.columns:
            rename = {**rename, **{k: v for k, v in
                                   [("identifier", "molecule_id")] if k in df.columns}}
        df = df.rename(columns=rename)
        if "molecule_id" not in df.columns:
            df["molecule_id"] = df.index.astype(str)
        # canonicalize the SMILES column name
        smi_col = next((c for c in df.columns if "smiles_source" in c or c.lower() == "smiles"), None)
        if smi_col and smi_col != "isomeric_SMILES":
            df = df.rename(columns={smi_col: "isomeric_SMILES"})
        if "isomeric_SMILES" not in df.columns:
            df["isomeric_SMILES"] = ""
        return df

    # ------------------------------------------------------------------
    # route prediction
    # ------------------------------------------------------------------
    _COLS = [
        "molecule_id", "target_smiles", "route_rank", "route_probability",
        "n_steps", "status", "reaction_smiles", "building_blocks",
        "building_blocks_availability",
    ]

    def _empty_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self._COLS)

    def predict_routes(self, targets: pd.DataFrame, availability: Dict[str, Any],
                       n_routes: int = 3) -> pd.DataFrame:
        """Return per-target route predictions (or empty frame if unavailable).

        Prefers running the standalone worker under the dedicated Python 3.12
        venv (aizynthfinder requires Python < 3.13); falls back to an
        in-process finder when importable.  Results reuse a single schema so
        downstream phases are agnostic to the staging.
        """
        if not availability.get("available"):
            return self._empty_frame()
        if not len(targets):
            self.log.warn("no targets to plan routes for")
            return self._empty_frame()

        if self.venv_python.exists():
            return self._predict_via_venv(targets, n_routes)
        finder = self._init_finder()
        if finder is None:
            return self._empty_frame()
        return self._predict_in_process(finder, targets, n_routes)

    def _predict_via_venv(self, targets: pd.DataFrame, n_routes: int) -> pd.DataFrame:
        """Run the standalone aizynth_worker under the 3.12 venv."""
        import json as _json
        import subprocess
        import tempfile
        payload = targets[["molecule_id", "isomeric_SMILES"]].to_dict("records")
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
                _json.dump(payload, fh)
                tgt = fh.name
            cfg = str(self.model_dir / "config.yml")
            proc = subprocess.run(
                [str(self.venv_python), str(self.cfg.root / "src" / "aizynth_worker.py"),
                 cfg, tgt],
                capture_output=True, text=True, timeout=3600,
            )
            if proc.returncode != 0:
                self.log.warn(f"aizynth worker exit {proc.returncode}: {proc.stderr[-500:]}")
                return self._empty_frame()
            out = _json.loads(proc.stdout)
            if out.get("error"):
                self.log.warn(f"aizynth worker error: {out['error'][-500:]}")
                return self._empty_frame()
            rows = out.get("rows", [])
        except Exception as e:  # pragma: no cover - depends on venv state
            self.log.warn(f"aizynth worker run failed: {e}")
            return self._empty_frame()
        finally:
            try:
                os.unlink(tgt)
            except Exception:
                pass
        df = pd.DataFrame(rows, columns=self._COLS)
        self.log.info(f"worker returned {len(df)} predicted/declared rows")
        return df

    def _predict_in_process(self, finder: Any, targets: pd.DataFrame,
                            n_routes: int) -> pd.DataFrame:
        """In-process prediction fallback (same schema as worker output)."""
        rows: List[Dict[str, Any]] = []
        for _, t in targets.iterrows():
            mid = str(t["molecule_id"])
            smi = str(t["isomeric_SMILES"]).strip()
            if not smi or smi.lower() in ("nan", "none"):
                rows.append({"molecule_id": mid, "target_smiles": smi,
                             "route_rank": 0, "route_probability": None,
                             "n_steps": None, "status": "NO_SMILES",
                             "reaction_smiles": "", "building_blocks": "",
                             "building_blocks_availability": ""})
                continue
            try:
                finder.target_smiles = smi
                finder.prepare_tree()
                finder.tree_search()
                finder.build_routes()
                finder.routes.compute_scores(*finder.scorers.objects())
                routes = list(finder.routes)[:n_routes]
                if not routes:
                    rows.append({"molecule_id": mid, "target_smiles": smi,
                                 "route_rank": 0, "route_probability": None,
                                 "n_steps": None, "status": "NO_ROUTE_FOUND",
                                 "reaction_smiles": "", "building_blocks": "",
                                 "building_blocks_availability": ""})
                    continue
                for rank, r in enumerate(routes, start=1):
                    try:
                        rt = r["reaction_tree"]
                        rxn = ".".join(x.reaction_smiles() for x in rt.reactions())
                        n_steps = len(list(rt.reactions()))
                        prob = float((r.get("score") or {}).get("state score") or 0.0)
                        bb = ";".join(sorted(set(str(l) for l in rt.leafs())))
                    except Exception:
                        rxn, n_steps, prob, bb = "", -1, 0.0, ""
                    rows.append({"molecule_id": mid, "target_smiles": smi,
                                 "route_rank": rank, "route_probability": prob,
                                 "n_steps": n_steps, "status": "PREDICTED_ROUTE",
                                 "reaction_smiles": rxn, "building_blocks": bb,
                                 "building_blocks_availability": "UNKNOWN"})
            except Exception as e:  # pragma: no cover - depends on external install
                rows.append({"molecule_id": mid, "target_smiles": smi,
                             "route_rank": 0, "route_probability": None,
                             "n_steps": None, "status": f"ERROR: {e}",
                             "reaction_smiles": "", "building_blocks": "",
                             "building_blocks_availability": ""})
        return pd.DataFrame(rows, columns=self._COLS)

    # ------------------------------------------------------------------
    # building-block summary (stock availability)
    # ------------------------------------------------------------------
    def _route_product_matches_target(self, reaction_smiles: Any,
                                      target_smi: Any) -> Optional[bool]:
        """True when the predicted route's product equals the target SMILES.

        Compares the (canonicalized) fragment before the first `>>` - the
        retrosynthetic final product - against the target.  Returns None when
        it cannot be determined (e.g. a degenerate multi-fragment route).
        """
        if not reaction_smiles or not target_smi:
            return None
        rxn = str(reaction_smiles)
        tgt = str(target_smi)
        if ">>" not in rxn:
            return None
        prod = rxn.split(">>")[0].split(".")[0].strip()
        for cand_target in (tgt,):
            for cand_prod in (prod,):
                try:
                    cp = Chem.MolToSmiles(Chem.MolFromSmiles(cand_prod))
                    ct = Chem.MolToSmiles(Chem.MolFromSmiles(cand_target))
                    if cp and ct:
                        return cp == ct
                except Exception:
                    continue
        return prod == tgt

    def building_block_report(self, routes: pd.DataFrame) -> pd.DataFrame:
        """Tabulate building-block coverage + stock availability per block."""
        cols = ["building_block", "n_targets", "in_stock", "stock_names"]
        if not len(routes) or "building_blocks" not in routes.columns:
            return pd.DataFrame(columns=cols)
        from collections import Counter
        cnt: Counter = Counter()
        stock_hit: Counter = Counter()  # b -> any route has this b stocked
        stock_names: dict = {}
        avail_col = routes["building_blocks_availability"] if \
            "building_blocks_availability" in routes.columns else None
        for i, row in routes.iterrows():
            bbs = str(row.get("building_blocks") or "").split(";")
            if not any(bbs):
                continue
            avs = str(avail_col.iloc[i] or "").split(";") if avail_col is not None else []
            for b in bbs:
                b = b.strip()
                if not b:
                    continue
                cnt[b] += 1
                # availability entries are aligned to the building-block list;
                # attempt position-based match
                idx = bbs.index(b) if b in bbs else -1
                av = avs[idx] if 0 <= idx < len(avs) else ""
                if av and av != "NOT_IN_STOCK" and av != "UNKNOWN":
                    stock_hit[b] += 1
                    stock_names.setdefault(b, av)
        rows = []
        for b, n in cnt.most_common():
            if stock_hit.get(b):
                rows.append({"building_block": b, "n_targets": n,
                             "in_stock": "IN_STOCK",
                             "stock_names": stock_names[b]})
            else:
                rows.append({"building_block": b, "n_targets": n,
                             "in_stock": "NOT_IN_STOCK", "stock_names": ""})
        return pd.DataFrame(rows, columns=cols)

    def run_round(self, round_no: int = 1) -> Dict[str, Any]:
        self.log.step(f"Synth route prediction for round {round_no}")
        availability = self.audit_availability()
        if not availability.get("available"):
            self.log.warn("AIZynthFinder/models NOT AVAILABLE - no route "
                          "predictions written (honest skips, no fabrication)")
        targets = self.collect_targets(round_no=round_no)
        routes = self.predict_routes(targets, availability)
        if len(routes):
            routes = routes.copy()
            routes["route_product_matches_target"] = [
                self._route_product_matches_target(pr, tr) if st == "PREDICTED_ROUTE" else None
                for pr, tr, st in zip(routes["reaction_smiles"],
                                      routes["target_smiles"],
                                      routes["status"])]
        bb = self.building_block_report(routes)
        # Count distinct targets with a predicted route (NOT route rows): a single
        # sensible target yields up to n_routes ranked rows, so row-counting would
        # overstate how many compounds are synthesizable.
        n_route_rows = int((routes["status"] == "PREDICTED_ROUTE").sum()) if len(routes) else 0
        n_targets_route = int(
            routes.loc[routes["status"] == "PREDICTED_ROUTE", "molecule_id"].nunique()
        ) if n_route_rows else 0
        clean = routes[(routes["status"] == "PREDICTED_ROUTE")
                       & (routes["route_product_matches_target"] == True)] if len(routes) else None
        n_targets_clean_route = int(
            clean["molecule_id"].nunique()) if clean is not None and len(clean) else 0
        save_df(routes, str(self.cfg.resolve("synth_route/synth_route_predictions.csv")))
        save_df(bb, str(self.cfg.resolve("synth_route/synth_route_building_blocks.csv")))
        manifest = {
            **make_provenance("synth_route", self.cfg, {}),
            "round": round_no,
            "availability": availability,
            "n_targets": int(len(targets)),
            "n_routes_predicted": n_targets_route,
            "n_predicted_route_rows": n_route_rows,
            "n_targets_with_clean_route": n_targets_clean_route,
            "n_targets_unrouted": int(len(targets)) - n_targets_route,
            "counting_note": ("n_routes_predicted counts distinct targets with at "
                              "least one PREDICTED_ROUTE (not route rows); "
                              "n_targets_with_clean_route further requires the "
                              "recorded route product to equal the target."),
            "honesty_note": ("Routes are PREDICTED_ROUTE / UNVERIFIED; they aid - "
                             "do not replace - the med-chem oversight gate."),
        }
        if not availability.get("available"):
            manifest["honesty_note"] = (
                "NOT AVAILABLE - AIZynthFinder and/or model files absent; no "
                "route predicted, nothing fabricated.")
        save_manifest(manifest, str(self.cfg.resolve("synth_route/synth_route_manifest.json")))
        report = write_report(self.cfg, {"routes": routes, "building_blocks": bb,
                                         "availability": availability,
                                         "n_targets": len(targets),
                                         "n_targets_with_clean_route": n_targets_clean_route})
        out_md = self.cfg.resolve("synth_route/synth_route_report.md")
        os.makedirs(os.path.dirname(str(out_md)), exist_ok=True)
        with open(str(out_md), "w", encoding="utf-8") as f:
            f.write(report)
        self.log.info(f"{n_route_rows} PREDICTED_ROUTE rows over "
                      f"{n_targets_route} targets; clean routes for "
                      f"{n_targets_clean_route} (targets={len(targets)})")
        return manifest


def _load_json_table(p: Path) -> pd.DataFrame:
    import json
    with open(str(p), "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        compounds = obj.get("compounds") or obj.get("candidates") or obj.get("molecules")
        if compounds is None:
            compounds = []
    else:
        compounds = obj
    return pd.DataFrame(compounds)


def write_report(cfg: ProjectConfig, res: Dict[str, Any]) -> str:
    availability = res.get("availability", {})
    routes = res.get("routes", pd.DataFrame())
    bb = res.get("building_blocks", pd.DataFrame())
    n_targets = res.get("n_targets", 0)
    n_pred_rows = int((routes["status"] == "PREDICTED_ROUTE").sum()) if len(routes) else 0
    n_pred_targets = int(
        routes.loc[routes["status"] == "PREDICTED_ROUTE", "molecule_id"].nunique()
    ) if n_pred_rows else 0
    # pass counts into the report via manifest keys carried in res when available
    n_clean = res.get("n_targets_with_clean_route", None)
    avail = "AVAILABLE" if availability.get("available") else "NOT AVAILABLE (auto-skip)"
    L = [
        "# SYNTH ROUTE PREDICTION REPORT",
        "",
        f"Status: {avail}",
        (f"Targets evaluated: {n_targets} | Targets with a predicted route: "
         f"{n_pred_targets}/{n_targets} ({n_pred_rows} PREDICTED_ROUTE rows: up to "
         f"3 ranked routes per target)"),
        "",
        "Honesty contract: every route below is PREDICTED_ROUTE / UNVERIFIED - a",
        "computer-generated plan from a pretrained retrosynthesis model, fed to the",
        "med-chem oversight gate as decision support. It is NOT a guarantee that",
        "the synthesis will work, and NO route is fabricated when the models are",
        "unavailable.",
        "",
    ]
    if availability.get("available") and len(routes):
        L += [
            "## Per-target route predictions (top 3)",
            "",
            routes.to_markdown(index=False) if hasattr(routes, "to_markdown") else routes.to_string(index=False),
            "",
        ]
        if len(bb):
            L += ["## Building-block availability", "", "",
                  bb.to_string(index=False)]
    else:
        L += ["No route predictions written (models absent)."]
    L.append("")
    return "\n".join(L)


def run_phase_synth_route(cfg_path: str) -> Dict[str, Any]:
    cfg = ProjectConfig(cfg_path)
    logger = ManagedLogger("synth_route", str(cfg.resolve("logs")))
    planner = SynthRoutePlanner(cfg, logger)
    return planner.run_round(round_no=1)


def main():  # pragma: no cover - CLI convenience
    import sys
    run_phase_synth_route(sys.argv[1] if len(sys.argv) > 1 else
                          os.environ.get("TAF_CONFIG",
                                         "/Users/nb/Documents/ligand-based-modeling/project/configs/config.yaml"))


if __name__ == "__main__":
    main()