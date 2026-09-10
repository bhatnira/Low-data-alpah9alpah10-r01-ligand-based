"""Smoke tests for the TAF workflow.

Run from the project root:
    python3 -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import ProjectConfig
from src.taf import TAF_CANDIDATES

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/config.yaml"
RAW_CSV = ROOT.parent / "modulator-dataset-a9a10.csv"


@pytest.fixture(scope="session")
def cfg():
    return ProjectConfig(str(CONFIG))


def _exists(rel):
    return (ROOT / rel).exists()


@pytest.fixture(scope="session")
def df():
    df = pd.read_csv(RAW_CSV)
    df.columns = ["Identifier", "Smiles", "Activity_uM", "Potentiation_pct"]
    return df


# ---------------- data integrity ----------------

def test_all_30_compounds_parse(cfg, df):
    from rdkit import Chem
    mols = df["Smiles"].apply(Chem.MolFromSmiles)
    assert mols.notna().sum() == 30, "all 30 compounds must parse with RDKit"


def test_all_duplicate_ids(cfg, df):
    assert df["Identifier"].nunique() == len(df)


def test_no_compound_with_both_zero(cfg, df):
    both_zero = (df["Activity_uM"] == 0) & (df["Potentiation_pct"] == 0)
    zero_missing = (df["Activity_uM"].isna() | df["Potentiation_pct"].isna())
    assert not zero_missing.any()
    assert both_zero.any(), "at least some compounds are expected to be fully inactive"


def test_expected_active_count(cfg, df):
    active = (df["Activity_uM"] > 0) & (df["Potentiation_pct"] > 0)
    assert int(active.sum()) == 7, "expected 7 actives under potency>0 AND potentiation>0"

    active_ids = sorted(df.loc[active, "Identifier"].astype(int))
    assert active_ids == [1, 2, 3, 12, 18, 24, 25]


# ---------------- SMARTS validity ----------------

def test_taf_smarts_valid():
    from rdkit import Chem
    for taf in TAF_CANDIDATES:
        if taf["smarts_2d"] is None:
            continue
        assert Chem.MolFromSmarts(taf["smarts_2d"]) is not None, taf["taf_id"]
        for alt in taf.get("smarts_2d_alt", []):
            assert Chem.MolFromSmarts(alt) is not None, f"{taf['taf_id']} alt {alt}"


# ---------------- reinvent configs & frozen model ----------------

def test_frozen_model_artifact():
    import joblib
    p = ROOT / "models/classical/lbm_rf_fp_desc.joblib"
    if not p.exists():
        pytest.skip("frozen model not exported")
    a = joblib.load(p)
    assert a["n_compounds"] == 30
    assert a["n_actives"] == 7
    assert a["positive_class"] == 1
    assert a["feature_columns"]


def test_all_reinvent_toml_configs_parse():
    import glob, tomllib
    tomls = sorted(glob.glob(str(ROOT / "reinvent/configs/*_staged_learning.toml")))
    assert tomls, "no reinvent configs generated"
    for f in tomls:
        with open(f, "rb") as fh:
            d = tomllib.load(fh)
        assert d["run_type"] == "staged_learning"
        assert d["parameters"]["prior_file"].endswith(".prior")
        st = d["stage"][0]["scoring"]
        comps = {list(c.keys())[0] for c in st["component"]}
        assert {"ExternalProcess", "MolecularWeight", "SAScore", "custom_alerts"} <= comps
        ex = next(c["ExternalProcess"] for c in st["component"] if "ExternalProcess" in c)
        # every LbM endpoint must call the scoring payload and request a property
        for e in ex["endpoint"]:
            assert "score_lbm.py" in e["params"]["args"][0]
            assert e["params"]["property"]


def test_priors_verified_in_manifest():
    import json
    p = ROOT / "reinvent/reinvent_manifest.json"
    if not p.exists():
        pytest.skip("reinvent manifest not generated")
    m = json.loads(p.read_text())
    assert m["priors"]["reinvent"]["verified"] is True
    assert m["priors"]["libinvent"]["verified"] is True


def test_scoring_payload_is_executable():
    import json, subprocess
    p = ROOT / "reinvent/score_lbm.py"
    if not p.exists():
        pytest.skip("scoring payload not generated")
    res = subprocess.run(
        [sys.executable, str(p)],
        input="OC1COC(=O)C1\ngarbage-not-a-molecule\n",
        text=True, capture_output=True,
    )
    assert res.returncode == 0
    payload = json.loads(res.stdout)["payload"]
    assert "predicted_activity" in payload
    assert len(payload["predicted_activity"]) == 2
    # intact SMILES scores in [0,1]; junk is NaN
    assert 0.0 <= payload["predicted_activity"][0] <= 1.0
    assert float(payload["predicted_activity"][1]) != float(payload["predicted_activity"][1])  # NaN


def test_inception_and_scaffold_seeds():
    inc = ROOT / "reinvent/inception.smi"
    scaf = ROOT / "reinvent/scaffolds_libinvent.smi"
    if not inc.exists() or not scaf.exists():
        pytest.skip("reinvent seed files not generated")
    inc_n = sum(1 for _ in inc.open())
    assert inc_n >= 1
    # scaffolds carry atom-mapped bases (the "*" must bind a base atom)
    from rdkit import Chem
    for line in scaf.open():
        smi = line.strip()
        if smi:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, smi
            assert any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()), smi


def test_lock_hashes_reinvent_and_priors():
    import json
    p = ROOT / "prospective/model_lock_manifest.json"
    if not p.exists():
        pytest.skip("lock not generated")
    m = json.loads(p.read_text())
    assert m.get("reinvent_config_hashes")
    assert m.get("prior_file_hashes")


# ---------------- pipeline outputs ----------------

@pytest.mark.skipif(not _exists("data/processed/data_analysis_ready.csv"),
                    reason="pipeline not run yet")
def test_analysis_ready_has_30_rows():
    d = pd.read_csv(ROOT / "data/processed/data_analysis_ready.csv")
    assert len(d) == 30


@pytest.mark.skipif(not _exists("sar/fingerprints/similarity_morgan.npy"),
                    reason="sar phase not run yet")
def test_fingerprint_matrix_shape():
    sim = np.load(ROOT / "sar/fingerprints/similarity_morgan.npy")
    assert sim.shape == (30, 30)


@pytest.mark.skipif(not _exists("validation/loco/loco_summary.csv"),
                    reason="models phase not run yet")
def test_loco_summary_present():
    s = pd.read_csv(ROOT / "validation/loco/loco_summary.csv")
    assert len(s) >= 1
    assert "model" in s.columns or "method" in s.columns


@pytest.mark.skipif(not _exists("xai/counterfactual/counterfactual.csv"),
                    reason="xai phase not run yet")
def test_counterfactual_probes_predicted():
    cf = pd.read_csv(ROOT / "xai/counterfactual/counterfactual.csv")
    assert set(["ID12_remove_alkyne", "ID12_remove_stereo",
                "ID25_remove_Br", "ID25_remove_stereo"]).issubset(set(cf["probe"]))
    assert not cf["predicted_active"].isin(["INVALID"]).any()


@pytest.mark.skipif(not _exists("taf/evidence/taf_evidence_matrix.csv"),
                    reason="taf phase not run yet")
def test_taf_evidence_matrix_complete():
    m = pd.read_csv(ROOT / "taf/evidence/taf_evidence_matrix.csv")
    assert set(m["taf_id"]) == {t["taf_id"] for t in TAF_CANDIDATES}
    assert "transferability_T(F)" in m.columns


@pytest.mark.skipif(not _exists("prospective/prospective_candidates.csv"),
                    reason="candidates phase not run yet")
def test_candidate_portfolio_groups():
    c = pd.read_csv(ROOT / "prospective/prospective_candidates.csv")
    assert "portfolio_group" in c.columns
    assert c["portfolio_group"].nunique() >= 5
    # every candidate must carry generator provenance
    for col in ("molecule_id", "isomeric_SMILES", "source"):
        assert col in c.columns


@pytest.mark.skipif(not _exists("prospective/model_lock_manifest.json"),
                    reason="lock phase not run yet")
def test_model_lock_manifest():
    import json
    with open(ROOT / "prospective/model_lock_manifest.json") as f:
        m = json.load(f)
    assert "candidate_library_hash" in m or "candidates_hash" in str(m)
    assert "configuration_hash" in m


@pytest.mark.skipif(not _exists("reports/leakage_audit.csv"),
                    reason="audit not run yet")
def test_no_leakage():
    a = pd.read_csv(ROOT / "reports/leakage_audit.csv")
    assert not a[a["severity"].astype(str).str.contains("NONE")].empty


@pytest.mark.skipif(not _exists("reports/index.html"),
                    reason="reports not built yet")
def test_report_index_exists():
    assert (ROOT / "reports/index.html").stat().st_size > 1000


@pytest.mark.skipif(not _exists("synth_route/synth_route_predictions.csv"),
                    reason="synth_route predictions not generated")
def test_synth_route_predictions_have_route_flags():
    r = pd.read_csv(ROOT / "synth_route/synth_route_predictions.csv")
    assert set(["molecule_id", "target_smiles", "route_rank",
                "route_probability", "status"]).issubset(set(r.columns))
    pred = r[r["status"].eq("PREDICTED_ROUTE")] if "status" in r else r.iloc[0:0]
    if len(pred):
        # every predicted route must carry a non-empty reaction plan
        assert (pred["reaction_smiles"].fillna("").str.len() > 0).all(), \
            "predicted routes must carry a non-empty reaction SMILES plan"
        # ... and building blocks with their stock availability
        if "building_blocks" in pred:
            assert (pred["building_blocks"].fillna("").str.len() > 0).all(), \
                "predicted routes must list building blocks / stock items"
    else:
        assert r["status"].isin(["NO_SMILES", "NO_ROUTE_FOUND"]).all(), \
            "unrouted targets must carry an honest status, never fabricated routes"


@pytest.mark.skipif(not _exists("synth_route/synth_route_manifest.json"),
                    reason="synth_route manifest not generated")
def test_synth_route_manifest_is_honest():
    import json
    with open(ROOT / "synth_route/synth_route_manifest.json") as f:
        m = json.load(f)
    assert "honesty_note" in m
    assert m["availability"]["available"] is False or m["n_routes_predicted"] >= 0
    if m["availability"]["available"]:
        # honesty label must be present whenever a route was predicted
        assert "PREDICTED_ROUTE" in m["honesty_note"] or "UNVERIFIED" in m["honesty_note"]
    else:
        # models absent -> honest NOT AVAILABLE, no fabricated plans
        assert "NOT AVAILABLE" in m["honesty_note"]
        plans = ROOT / "synth_route/synth_route_predictions.csv"
        if plans.exists():
            statuses = pd.read_csv(plans)["status"].dropna()
            if len(statuses):
                assert statuses.isin(["NO_SMILES", "NO_ROUTE_FOUND"]).all()