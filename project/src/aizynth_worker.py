#!/usr/bin/env python3
"""Standalone AIZynthFinder route-prediction worker (runs inside the
dedicated Python 3.12 venv: project/venvs/aizynth312).

The main pipeline runs on Python 3.14 but aizynthfinder 4.4.1 requires
Python < 3.13, so this worker is invoked as a subprocess by
src/synth_route.py passing a JSON payload on stdin and reading the
prediction rows back on stdout.

Usage:
    aizynth_worker.py <config.yml> <targets.json>

targets.json:  [{"molecule_id": "MC-6", "isomeric_SMILES": "..."}, ...]
stdout JSON:   {"rows": [{...prediction..., }, ...]}
"""  # pragma: no cover - executed as a subprocess

import json
import sys
import traceback

# ----------------------------------------------------------------------
# route extraction helpers
# ----------------------------------------------------------------------
def _route_to_row(mid: str, smi: str, rank: int, route_dict, finder) -> dict:
    rt = route_dict.get("reaction_tree")
    score = (route_dict.get("score") or {}).get("state score")
    if score is None:
        score = (route_dict.get("all_scores") or {}).get("state score")
    solved = bool((route_dict.get("route_metadata") or {}).get("is_solved"))
    n_steps, reaction_smiles = -1, ""
    bb_stock_map: dict = {}
    if rt is not None:
        try:
            reaction_smiles = ".".join(
                rxn.reaction_smiles() for rxn in rt.reactions()
            )
            n_steps = len(list(rt.reactions()))
        except Exception:
            n_steps, reaction_smiles = -1, ""
        try:
            for leaf in rt.leafs():
                bb_stock_map.setdefault(str(leaf), [])
                avail = list(finder.stock.availability_list(leaf)) if finder else []
                if avail:
                    bb_stock_map[str(leaf)] = sorted(set(avail))
        except Exception:
            bb_stock_map = {}
    building_blocks = sorted(bb_stock_map.keys())
    bb_stock = [";".join(v) if v else "NOT_IN_STOCK" for v in
                (bb_stock_map[b] for b in building_blocks)]
    return {
        "molecule_id": mid,
        "target_smiles": smi,
        "route_rank": rank,
        "route_probability": float(score) if score is not None else None,
        "n_steps": n_steps,
        "status": "PREDICTED_ROUTE" if solved else "NO_ROUTE_FOUND",
        "reaction_smiles": reaction_smiles,
        "building_blocks": ";".join(building_blocks),
        "building_blocks_availability": ";".join(bb_stock),
    }


def predict_for_targets(finder, targets, n_routes: int) -> list:
    rows = []
    for t in targets:
        mid = str(t.get("molecule_id"))
        smi = str(t.get("isomeric_SMILES", "")).strip()
        if not smi or smi.lower() in ("nan", "none"):
            rows.append({
                "molecule_id": mid, "target_smiles": smi,
                "route_rank": 0, "route_probability": None, "n_steps": None,
                "status": "NO_SMILES", "reaction_smiles": "", "building_blocks": "",
                "building_blocks_availability": "",
            })
            continue
        try:
            finder.target_smiles = smi
            finder.prepare_tree()
            finder.tree_search()
            finder.build_routes()
            finder.routes.compute_scores(*finder.scorers.objects())
            routes = list(finder.routes)[:n_routes]
            if not routes:
                rows.append({
                    "molecule_id": mid, "target_smiles": smi,
                    "route_rank": 0, "route_probability": None, "n_steps": None,
                    "status": "NO_ROUTE_FOUND",
                    "reaction_smiles": "", "building_blocks": "",
                    "building_blocks_availability": "",
                })
                continue
            for rank, r in enumerate(routes, start=1):
                rows.append(_route_to_row(mid, smi, rank, r, finder))
        except Exception as exc:  # pragma: no cover - defensive
            rows.append({
                "molecule_id": mid, "target_smiles": smi,
                "route_rank": 0, "route_probability": None, "n_steps": None,
                "status": f"ERROR: {exc}", "reaction_smiles": "", "building_blocks": "",
                "building_blocks_availability": "",
            })
    return rows


def main() -> int:
    if len(sys.argv) != 3:
        print(json.dumps({"error": "usage: aizynth_worker.py <config.yml> <targets.json>"}))
        return 2
    config_file, targets_file = sys.argv[1], sys.argv[2]
    with open(targets_file, "r", encoding="utf-8") as fh:
        targets = json.load(fh)
    n_routes = 3
    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder
        finder = AiZynthFinder(configfile=config_file)
        finder.stock.select(finder.stock.items)
        finder.expansion_policy.select(finder.expansion_policy.items[0])
        finder.filter_policy.select_all()
        rows = predict_for_targets(finder, targets, n_routes)
        print(json.dumps({"rows": rows}, default=str))
        return 0
    except Exception:
        print(json.dumps({"error": traceback.format_exc(), "rows": []}, default=str))
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())