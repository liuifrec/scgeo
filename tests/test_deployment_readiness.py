"""Read-only contracts and a small-n design; no biological validation claims."""
import copy
import json
from pathlib import Path
import runpy
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scgeo as sg


ROOT = Path(__file__).resolve().parents[1]
SMOKE = runpy.run_path(str(ROOT / "examples/deployment_smoke.py"))
PREFLIGHT = runpy.run_path(str(ROOT / "scripts/deployment_preflight.py"))


def codes(report):
    return {c["code"] for c in report["checks"] if c["severity"] == "error"}


def test_contract_is_read_only_and_keeps_insufficient_coverage(monkeypatch):
    data, config = SMOKE["synthetic_design"]()
    before = data.copy()
    before_config = copy.deepcopy(config)
    def forbidden(*args, **kwargs):
        raise AssertionError("Validation must not estimate or fit")
    for name in ("robust_shift", "representation_stability", "local_geometry_stability"):
        monkeypatch.setattr(sg.tl, name, forbidden)
    report = sg.get.input_contract(data, config=config)
    assert report["ok"] and report["sample_inference_eligible"]
    assert report["n_samples_per_condition"] == [2, 2]
    assert report["eligible_states"] == ["shared"]
    rows = {r["state"]: r for r in report["state_coverage"]}
    assert rows["sparse"]["status"] == "insufficient_samples"
    assert rows["control_only"]["status"] == "missing_condition"
    assert rows["tiny"]["status"] == "insufficient_cells"
    assert rows["sparse"]["sample_counts"][-1]["n_cells"] == 0
    assert "scgeo" not in data.uns
    assert config == before_config
    pd.testing.assert_frame_equal(data.obs, before.obs)
    for key in data.obsm:
        np.testing.assert_array_equal(data.obsm[key], before.obsm[key])
    assert data.uns["synthetic_projection"] == before.uns["synthetic_projection"]
    np.testing.assert_array_equal(data.uns["paga"]["connectivities"], before.uns["paga"]["connectivities"])
    assert data.X is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("update,code", [
    ({"sample_key": None}, "sample_design"),
    ({"bootstrap_unit": "cell"}, "bootstrap"),
    ({"design": "paired"}, "sample_design"),
    ({"min_samples_per_condition": 1}, "sample_threshold"),
    ({"reps": ["X_primary"]}, "representations"),
    ({"reps": ["X_primary", "not_present"]}, "matrix_missing"),
    ({"reps": "X_primary"}, "config"),
    ({"embedding_key": "display"}, "matrix_missing"),
    ({"group0": "absent"}, "groups"),
    ({"group1": "control"}, "groups"),
    ({"min_cells": 0}, "threshold"),
    ({"min_cells": 1.5}, "threshold"),
    ({"methods": ["invented"]}, "methods"),
    ({"k_values": [1000]}, "local_k"),
    ({"k_values": [True]}, "local_k"),
    ({"state_key": []}, "config"),
    ({"projected_velocity": {"key": []}}, "velocity_config"),
    ({"evidence": {"robust_shift": {"method": "robust_shift", "params": {"sample_key": "sample"}}}}, "evidence_missing"),
    ({"typo": "never ignored"}, "config"),
])
def test_invalid_config_is_reported_without_substitution(update, code):
    data, config = SMOKE["synthetic_design"]()
    config.update(update)
    report = sg.get.input_contract(data, config=config)
    assert not report["ok"] and not report["sample_inference_eligible"]
    assert code in codes(report)
    assert "scgeo" not in data.uns


@pytest.mark.parametrize("defect,code", [
    ("nonfinite", "nonfinite"), ("missing", "missing_labels"),
    ("collision", "label_collision"), ("sample_mix", "sample_condition"),
    ("velocity_shape", "velocity_shape"), ("velocity_provenance", "velocity_provenance"),
    ("graph_order", "paga"), ("graph_negative", "paga"), ("graph_provenance", "paga"),
    ("duplicate_obs", "observation_identity"),
    ("sparse_display", "matrix_type"), ("complex_rep", "matrix_type"),
    ("invalid_labels", "labels"),
])
def test_input_defects_are_reported(defect, code):
    data, config = SMOKE["synthetic_design"]()
    if defect == "nonfinite":
        data.obsm["X_primary"][0, 0] = np.nan
    elif defect == "missing":
        data.obs.loc[data.obs.index[0], "sample"] = None
    elif defect == "collision":
        data.obs["state"] = data.obs["state"].astype(object)
        data.obs.loc[data.obs.index[:2], "state"] = [1, "1"]
    elif defect == "sample_mix":
        data.obs.loc[data.obs.condition.eq("treated"), "sample"] = "sample_0"
    elif defect == "velocity_shape":
        data.obsm["velocity_display"] = np.zeros((data.n_obs, 3))
    elif defect == "velocity_provenance":
        data.uns["synthetic_projection"]["basis"] = "X_other"
    elif defect == "graph_order":
        data.uns["paga"]["node_order"] = list(reversed(config["paga"]["node_order"]))
    elif defect == "graph_negative":
        data.uns["paga"]["connectivities"][0, 1] = -1
    elif defect == "graph_provenance":
        del data.uns["paga"]["source"]
    elif defect == "duplicate_obs":
        data.obs.index = ["duplicate"] * data.n_obs
    elif defect == "sparse_display":
        data.obsm["X_display"] = sparse.csr_matrix(data.obsm["X_display"])
    elif defect == "complex_rep":
        data.obsm["X_primary"] = data.obsm["X_primary"].astype(complex) + 1j
    elif defect == "invalid_labels":
        data.obs["state"] = [["invalid"] for _ in range(data.n_obs)]
    report = sg.get.input_contract(data, config=config)
    assert code in codes(report)
    assert not report["ok"]


def test_descriptive_mode_does_not_claim_sample_inference():
    data, config = SMOKE["synthetic_design"]()
    config.update(purpose="descriptive", sample_key=None, methods=["plotting"], bootstrap_unit="auto")
    report = sg.get.input_contract(data, config=config)
    assert report["ok"] and not report["sample_inference_eligible"]
    assert report["eligible_states"] == []


def test_sparse_coordinates_without_expression_access():
    data, config = SMOKE["synthetic_design"]()
    data.obsm["X_primary"] = sparse.csr_matrix(data.obsm["X_primary"])
    assert sg.get.input_contract(data, config=config)["ok"]
    assert data.X is None


def test_four_sample_estimands_stability_and_report_contracts():
    data, config = SMOKE["synthetic_design"]()
    table = SMOKE["analyze"](data).set_index("state")
    results = data.uns["scgeo"]
    shifts = results["robust_shift"]
    np.testing.assert_allclose(shifts["by"]["shared"]["delta"], [1., 0.], atol=1e-6)
    # Unequal cell counts must not change equal weighting of independent samples.
    cell_delta = (data.obsm["X_primary"][(data.obs.state == "shared") & (data.obs.condition == "treated")].mean(0)
                  - data.obsm["X_primary"][(data.obs.state == "shared") & (data.obs.condition == "control")].mean(0))
    assert not np.isclose(cell_delta[0], 1., atol=1e-3)
    assert shifts["params"]["resolved_bootstrap_unit"] == "sample"
    per_rep = results["representation_stability"]["per_rep_state"]
    assert set(per_rep.loc[per_rep.node == "sparse", "status"]) == {"underpowered_samples"}
    assert not per_rep.loc[per_rep.node != "shared", "usable"].any()
    assert table.loc["sparse", "descriptive_only"]
    assert table.loc["control_only", "descriptive_only"]
    # Report inference_level describes resampling units, not all coverage gates.
    assert table.loc["tiny", "inference_level"] == "biological_sample"
    assert table.loc["tiny", "coverage_status"] == "insufficient_coverage"
    contract = sg.get.input_contract(data, config=config)
    assert "tiny" in contract["excluded_states"]
    enum = SMOKE["exact_assignments"](data)
    assert enum["n_assignments"] == 6
    assert len({tuple(x["treated_samples"]) for x in enum["assignments"]}) == 6
    assert sum(np.isclose(x["delta_norm"], 1.) for x in enum["assignments"]) == 2
    config["evidence"] = {"robust_shift": {"method": "robust_shift", "params": {"sample_key": "sample", "rep": "X_primary"}}}
    assert sg.get.input_contract(data, config=config)["ok"]
    config["evidence"]["robust_shift"]["params"]["sample_key"] = "other"
    assert "evidence_mismatch" in codes(sg.get.input_contract(data, config=config))
    del results["robust_shift"]["global"]
    assert "evidence_schema" in codes(sg.get.input_contract(data, config=config))


@pytest.fixture
def source_checkout(tmp_path):
    root = tmp_path / "source"
    (root / "scgeo").mkdir(parents=True)
    (root / "scgeo/__init__.py").write_text('__version__ = "0.1.0"\n')
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE).decode().strip()
    git("init")
    git("add", "scgeo")
    git("-c", "user.name=ScGeo test", "-c", "user.email=synthetic@example.invalid", "commit", "-m", "synthetic source")
    return root, git("rev-parse", "HEAD")


def test_preflight_checks_imported_bytes_not_version_string(source_checkout, tmp_path):
    root, commit = source_checkout
    verify = PREFLIGHT["_source_report"]
    assert verify(root, commit, root / "scgeo/__init__.py")["verified"]
    published = tmp_path / "site-packages/scgeo"
    published.mkdir(parents=True)
    published.joinpath("__init__.py").write_text('__version__ = "0.1.0"\n# older published implementation\n')
    result = verify(root, commit, published / "__init__.py")
    assert not result["verified"]
    assert result["mismatched_files"] == ["scgeo/__init__.py"]
    published.joinpath("__init__.py").write_bytes((root / "scgeo/__init__.py").read_bytes())
    assert verify(root, commit, published / "__init__.py")["verified"]
    published.joinpath("unexpected.py").write_text("# untracked extension\n")
    assert not verify(root, commit, published / "__init__.py")["verified"]
    with pytest.raises(ValueError, match="full"):
        verify(root, "main", published / "__init__.py")
    root.joinpath("uncommitted.txt").write_text("preserve me")
    result = verify(root, commit, root / "scgeo/__init__.py")
    assert result["dirty"] and not result["verified"]
    subprocess.run(["git", "-C", str(root), "add", "uncommitted.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=ScGeo test", "-c",
                    "user.email=synthetic@example.invalid", "commit", "-m", "later source"], check=True, capture_output=True)
    result = verify(root, commit, root / "scgeo/__init__.py")
    assert not result["dirty"] and result["commit"] != commit and not result["verified"]


def test_environment_report_records_actual_import_and_missing_optionals(monkeypatch, source_checkout):
    root, commit = source_checkout
    function = PREFLIGHT["environment_report"]
    monkeypatch.setattr(PREFLIGHT["importlib"].util, "find_spec", lambda name: None)
    report = function(expected_commit=commit, source_root=root, require_optional=["unknown"], pip_check=False)
    assert not report["ok"]
    assert report["scgeo"]["file"] == str(Path(sg.__file__).resolve())
    assert report["python"]["executable"]
    assert report["packages"] and not report["optional"]["scanpy"]["available"]
    assert any("Unknown optional" in x for x in report["errors"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("broken", ["scgeo", "scvelo"])
def test_environment_import_failures_are_actionable(monkeypatch, source_checkout, broken):
    root, commit = source_checkout
    real_import = PREFLIGHT["importlib"].import_module
    def broken_import(name, *args, **kwargs):
        if name == broken:
            raise ImportError("synthetic missing dependency")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(PREFLIGHT["importlib"], "import_module", broken_import)
    report = PREFLIGHT["environment_report"](
        expected_commit=commit, source_root=root, require_optional=["scvelo"], pip_check=False)
    assert not report["ok"]
    assert any("synthetic missing dependency" in message for message in report["errors"])
