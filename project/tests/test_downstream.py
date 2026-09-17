"""Tests for the generated-library downstream triage (src/downstream.py).

Verifies MASTER_PROMPT sections 14/15/30/33/34-47/48-53/61/67 runtime outputs.
Run from the project root:
    python3 -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import ProjectConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/config.yaml"


@pytest.fixture(scope="session")
def cfg():
    return ProjectConfig(str(CONFIG))


def _p(rel):
    return ROOT / rel


def test_generated_predictions_exists_and_schema(cfg):
    p = _p("generated/generated_predictions.csv")
    if not p.exists():
        pytest.skip("downstream phase not yet run")
    df = pd.read_csv(p)
    assert len(df) > 0
    required = ["molecule_id", "isomeric_SMILES", "predicted_activity",
                "predictive_uncertainty", "applicability_domain", "novelty",
                "confidence_class", "information_gain_score"]
    assert set(required) <= set(df.columns)
    assert df["applicability_domain"].isin(
        ["IN_DOMAIN", "BORDERLINE", "OUT_OF_DOMAIN", "UNKNOWN"]).all()
    assert df["confidence_class"].isin(["HIGH", "MODERATE", "LOW", "UNKNOWN"]).all()


def test_generated_qc_counts_nonnegative_and_real(cfg):
    """QC counts (canonical record) agree between CSV artifact and manifest provenance."""
    p = _p("generated/generated_qc.csv")
    if not p.exists():
        pytest.skip("downstream phase not yet run")
    import json
    qc = pd.read_csv(p).iloc[0].to_dict()
    for k in ["N_generated", "N_valid", "N_unique", "N_stereo_valid", "N_duplicate"]:
        assert k in qc and qc[k] >= 0
    assert qc["N_generated"] > 0
    assert qc["N_valid"] <= qc["N_generated"]
    assert qc["N_unique"] <= qc["N_generated"]
    mp = _p("generated/generated_qc.json")
    if mp.exists():
        prov = json.loads(mp.read_text())
        assert int(prov["input_hashes"]["N_generated"]) == int(qc["N_generated"])


def test_no_predicted_label_fabrication(cfg):
    """All generated rows must stay EXPERIMENTAL severity: untested/prospective (sect. 56)."""
    p = _p("reinvent/generated_molecules.csv")
    if not p.exists():
        pytest.skip("no generated molecules")
    df = pd.read_csv(p)
    assert (df["experimental_status"] == "untested").all()


def test_selectivity_never_fabricated(cfg):
    """Without subtype data, selectivity must be UNKNOWN (sect. 46-47)."""
    p = _p("selectivity/selectivity_final_filter.csv")
    if not p.exists():
        pytest.skip("downstream phase not yet run")
    df = pd.read_csv(p)
    assert set(df["selectivity_status"].unique()) <= {"UNKNOWN", "NOT_EVALUABLE"}
    assert (df["selectivity_uncertainty"] == "HIGH").any()


def test_self_similarity_audit(cfg):
    """Applicability domain must be LOO (no self-similarity) (sect. 14).

    The reported min_distance_to_any must equal the min distance to the OTHER
    training compounds (diagonal excluded). Before the LOO fix every row was 0
    (self-similarity contamination); after the fix, only genuine near-duplicate
    scaffolds may reach 0.
    """
    p = _p("validation/applicability_domain.csv")
    sim_path = _p("sar/fingerprints/similarity_morgan.npy")
    if not (p.exists() and sim_path.exists()):
        pytest.skip("models phase not yet run")
    import numpy as np
    df = pd.read_csv(p)
    sim = np.load(sim_path)
    assert len(df) == sim.shape[0]
    expected = []
    for i in range(sim.shape[0]):
        mask = np.ones(sim.shape[0], dtype=bool)
        mask[i] = False
        expected.append(1 - sim[i, mask].max())
    got = df["min_distance_to_any"].fillna(np.nan)
    assert np.allclose(got.round(4), np.round(expected, 4), equal_nan=True), (
        "applicability-domain distances are not leave-one-out")
    assert df["applicability_domain"].isin(
        ["IN_DOMAIN", "BORDERLINE", "OUT_OF_DOMAIN", "UNKNOWN"]).all()
    # A near-duplicate pair (L-ascorbate vs its undefined-stereo analog) may give
    # distance 0 to each other, but MORE than half the compounds must have a
    # strictly positive LOO distance (i.e. the old always-0 diagonal bug is gone).
    assert (got > 0).sum() >= len(df) // 2


def test_final_portfolio_schema(cfg):
    p = _p("portfolio/final_candidate_portfolio.csv")
    if not p.exists():
        pytest.skip("downstream phase not yet run")
    df = pd.read_csv(p)
    assert len(df) > 0
    required = ["molecule_id", "isomeric_SMILES", "parent_id", "generation_mode",
                "generation_objective", "experimental_parent", "predicted_potency",
                "potency_uncertainty", "applicability_domain", "novelty",
                "synthetic_feasibility", "cochlear_delivery", "systemic_ADME",
                "systemic_toxicity", "local_cochlear_safety", "selectivity",
                "selectivity_uncertainty", "information_gain", "portfolio_tier",
                "confidence", "selection_rationale", "risk_flags"]
    assert set(required) <= set(df.columns)
    assert set(df["portfolio_tier"].unique()) <= {
        "Tier1_high_priority", "Tier2_mechanistically_informative",
        "Tier3_exploratory", "EXCLUDE_OR_HOLD"}


def test_structure_based_handoff_exists(cfg):
    p = _p("structure_based_handoff.csv")
    if not p.exists():
        pytest.skip("downstream phase not yet run")
    df = pd.read_csv(p)
    assert len(df) > 0
    assert "no_receptor_structure_assumption" in df.columns
    assert (df["no_receptor_structure_assumption"] == "TRUE (ligand-based pipeline)").all()


def test_generated_library_integration_traceable(cfg):
    """Every portfolio molecule must trace back to the generated library (sect. 67)."""
    lib = _p("reinvent/generated_molecules.csv")
    port = _p("portfolio/final_candidate_portfolio.csv")
    if not (lib.exists() and port.exists()):
        pytest.skip("artifacts not yet produced")
    lib_ids = set(pd.read_csv(lib)["molecule_id"])
    port_ids = set(pd.read_csv(port)["molecule_id"])
    assert port_ids <= lib_ids


def test_no_invalid_molecule_in_pipeline(cfg):
    """Downstream scoring must only process valid, parseable SMILES."""
    from rdkit import Chem
    p = _p("generated/generated_predictions.csv")
    if not p.exists():
        pytest.skip("downstream phase not yet run")
    df = pd.read_csv(p)
    bad = df["isomeric_SMILES"].map(Chem.MolFromSmiles).isna().sum()
    assert bad == 0