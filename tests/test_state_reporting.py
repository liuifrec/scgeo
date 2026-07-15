import copy
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def _make_reporting_adata(*, n_per_condition=24):
    rng = np.random.RandomState(101)
    specs = [
        ("moved", np.array([0.0, 0.0]), np.array([1.2, 0.0]), np.array([1.0, 0.0]), n_per_condition),
        ("discordant", np.array([4.0, 0.0]), np.array([1.0, 0.0]), np.array([-1.0, 0.0]), n_per_condition),
        ("null", np.array([0.0, 4.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), n_per_condition),
        ("tiny", np.array([4.0, 4.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0]), 4),
    ]
    rows = []
    velocities = []
    obs_rows = []
    for state, base, delta, velocity, n in specs:
        for condition, offset in [("A", np.zeros(2)), ("B", delta)]:
            cloud = base + offset + rng.normal(scale=0.035, size=(n, 2))
            rows.append(cloud)
            velocities.append(np.repeat(velocity[None, :], n, axis=0))
            for i in range(n):
                obs_rows.append(
                    {
                        "state": state,
                        "condition": condition,
                        "sample": f"{condition}_s{i % 3}",
                    }
                )
    X = np.vstack(rows).astype(np.float32)
    V = np.vstack(velocities).astype(np.float32)
    obs = pd.DataFrame(obs_rows, index=[f"c{i}" for i in range(X.shape[0])])
    adata = ad.AnnData(X=np.zeros((X.shape[0], 1)), obs=obs)
    rot = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    adata.obsm["X_base"] = X
    adata.obsm["X_rot"] = (X @ rot).astype(np.float32)
    adata.obsm["X_umap"] = X + rng.normal(scale=0.01, size=X.shape)
    adata.obsm["V_base"] = V
    adata.obsm["V_rot"] = (V @ rot).astype(np.float32)
    return adata


def _snapshot_adata(adata):
    return {
        "obs": adata.obs.copy(deep=True),
        "uns_keys": sorted(adata.uns.keys()),
        "scgeo_keys": sorted(adata.uns.get("scgeo", {}).keys()),
        "obsm": {key: np.asarray(value).copy() for key, value in adata.obsm.items()},
    }


def _assert_adata_snapshot_equal(adata, before):
    pd.testing.assert_frame_equal(adata.obs, before["obs"])
    assert sorted(adata.uns.keys()) == before["uns_keys"]
    assert sorted(adata.uns.get("scgeo", {}).keys()) == before["scgeo_keys"]
    for key, value in before["obsm"].items():
        np.testing.assert_allclose(np.asarray(adata.obsm[key]), value)


def _run_full_report_stack(adata, *, store_per_cell=True):
    import scgeo as sg

    sg.tl.robust_shift(
        adata,
        rep="X_base",
        condition_key="condition",
        group0="A",
        group1="B",
        by="state",
        sample_key="sample",
        center="mean",
        n_boot=4,
        seed=2,
    )
    sg.tl.representation_stability(
        adata,
        reps=["X_base", "X_rot"],
        node_key="state",
        condition_key="condition",
        group0="A",
        group1="B",
        sample_key="sample",
        center="mean",
        n_boot=2,
        velocity_keys={"X_base": "V_base", "X_rot": "V_rot"},
        min_cells=10,
        seed=3,
    )
    sg.tl.local_geometry_stability(
        adata,
        reps=["X_base", "X_rot"],
        node_key="state",
        sample_key="sample",
        k_values=(5,),
        n_boot=2,
        max_exact_cells=200,
        store_per_cell=store_per_cell,
        seed=4,
    )
    return adata


def test_state_report_full_modules_reason_codes_summary_and_order():
    import scgeo as sg

    adata = _run_full_report_stack(_make_reporting_adata())
    report = sg.get.state_report(adata, node_key="state")

    assert report["state"].tolist() == ["moved", "discordant", "null", "tiny"]
    required = {
        "state",
        "normalized_delta_norm",
        "magnitude_ci95_low",
        "directional_stability",
        "representation_coverage_count",
        "representation_consensus_label",
        "magnitude_rank_median",
        "median_neighbor_overlap",
        "median_local_shape_distortion",
        "worst_local_shape_distortion",
        "local_k_values",
        "local_n_valid_pairs",
        "median_alignment_cosine",
        "bootstrap_unit",
        "inference_level",
        "descriptive_only",
        "effect_status",
        "stability_status",
        "local_geometry_status",
        "dynamics_status",
        "reason_codes",
        "summary",
    }
    assert required <= set(report.columns)

    moved = report.set_index("state").loc["moved"]
    assert "large_robust_shift" not in moved["reason_codes"]
    assert "stable_across_representations" in moved["reason_codes"]
    assert "neighborhoods_preserved" not in moved["reason_codes"]
    assert "dynamics_aligned" in moved["reason_codes"]
    assert "Normalized displacement" in moved["summary"]
    assert moved["bootstrap_unit"] == "sample"
    assert moved["inference_level"] == "biological_sample"
    assert not bool(moved["descriptive_only"])

    discordant = report.set_index("state").loc["discordant"]
    assert "dynamics_discordant" in discordant["reason_codes"]

    tiny = report.set_index("state").loc["tiny"]
    assert "insufficient_coverage" in tiny["reason_codes"]
    assert report["state_graph_agreement"].isna().all()
    assert "state_graph_summary" in report.attrs["global_diagnostics"]
    assert report.attrs["provenance"]["comparison_label"] == "B_vs_A"
    assert report.attrs["scgeo_report_rules"]["consensus_label_rules"] is not None

    report2 = sg.get.state_report(adata, node_key="state")
    report_no_attrs = report.copy()
    report2_no_attrs = report2.copy()
    report_no_attrs.attrs = {}
    report2_no_attrs.attrs = {}
    pd.testing.assert_frame_equal(report_no_attrs, report2_no_attrs)


def test_state_report_robust_shift_only_missing_modules_and_strict_mode():
    import scgeo as sg

    adata = _make_reporting_adata()
    sg.tl.robust_shift(
        adata,
        rep="X_base",
        condition_key="condition",
        group0="A",
        group1="B",
        by="state",
        center="mean",
        n_boot=0,
        seed=1,
    )

    report = sg.get.state_report(adata, node_key="state", strict=False)
    assert set(report["state"]) == {"moved", "discordant", "null", "tiny"}
    assert report["warnings"].str.contains("representation_stability result not found").any()
    assert report["representation_consensus_label"].isna().all()
    assert np.isfinite(report.loc[report["state"] == "moved", "delta_norm"]).all()
    moved = report.set_index("state").loc["moved"]
    assert moved["bootstrap_unit"] == "cell"
    assert moved["inference_level"] == "cell_descriptive"
    assert bool(moved["descriptive_only"])

    with pytest.raises(KeyError, match="representation_stability result not found"):
        sg.get.state_report(adata, node_key="state", strict=True)


def test_state_report_missing_velocity_and_one_representation():
    import scgeo as sg

    adata = _make_reporting_adata()
    sg.tl.robust_shift(
        adata,
        rep="X_base",
        condition_key="condition",
        group0="A",
        group1="B",
        by="state",
        center="mean",
        n_boot=0,
        seed=1,
    )
    sg.tl.representation_stability(
        adata,
        reps=["X_base"],
        node_key="state",
        condition_key="condition",
        group0="A",
        group1="B",
        center="mean",
        n_boot=0,
        velocity_keys={"X_base": None},
        min_cells=10,
        seed=1,
    )
    report = sg.get.state_report(adata, node_key="state")
    moved = report.set_index("state").loc["moved"]
    assert moved["representation_consensus_label"] == "insufficient_coverage"
    assert "dynamics_unavailable" in moved["reason_codes"]
    assert "insufficient_coverage" in moved["reason_codes"]


def test_state_evidence_panel_returns_figure_and_omits_absent_tracks():
    import matplotlib.figure
    import scgeo as sg

    adata = _make_reporting_adata()
    sg.tl.robust_shift(
        adata,
        rep="X_base",
        condition_key="condition",
        group0="A",
        group1="B",
        by="state",
        center="mean",
        n_boot=0,
        seed=1,
    )
    report = sg.get.state_report(adata, node_key="state")
    fig, data = sg.pl.state_evidence_panel(report, return_data=True, show=False)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert "Effect" in data["tracks"]
    assert "Stability" not in data["tracks"]
    assert "Local preservation" not in data["tracks"]
    assert "Dynamics" not in data["tracks"]


def test_representation_heatmap_dimensions_annotations_and_consensus_map_mutation():
    import matplotlib.figure
    import scgeo as sg

    adata = _run_full_report_stack(_make_reporting_adata())
    before_consensus = adata.uns["scgeo"]["representation_stability"]["consensus_state"].copy(deep=True)
    before_uns_keys = copy.deepcopy(sorted(adata.uns["scgeo"].keys()))

    heatmap, data = sg.pl.representation_stability_heatmap(
        adata,
        return_data=True,
        show=False,
    )
    assert isinstance(heatmap, matplotlib.figure.Figure)
    assert data["matrix"].shape == (4, 2)
    assert "consensus_label" in data["annotations"]
    assert set(data["representation_diagnostics"].index) == {"X_base", "X_rot"}

    before = _snapshot_adata(adata)
    cmap_fig = sg.pl.consensus_state_map(adata, node_key="state", show=False)
    assert isinstance(cmap_fig, matplotlib.figure.Figure)
    assert "Display embedding only" in cmap_fig.axes[0].get_title()
    colors = cmap_fig.axes[0].collections[0].get_facecolors()
    assert np.unique(colors[:, :3], axis=0).shape[0] >= 3
    _assert_adata_snapshot_equal(adata, before)

    after_consensus = adata.uns["scgeo"]["representation_stability"]["consensus_state"]
    pd.testing.assert_frame_equal(before_consensus, after_consensus)
    assert before_uns_keys == sorted(adata.uns["scgeo"].keys())


def test_perturbation_report_save_dir_and_local_distortion_map(tmp_path):
    import matplotlib.figure
    import scgeo as sg

    adata = _run_full_report_stack(_make_reporting_adata(), store_per_cell=True)
    bundle = sg.pl.perturbation_report(
        adata,
        node_key="state",
        save_dir=tmp_path,
        prefix="synthetic",
        comparison_label="A_to_B",
        show=False,
    )
    assert "state_report_csv" in bundle["saved_paths"]
    assert "representation_diagnostics_csv" in bundle["saved_paths"]
    assert "metadata_json" in bundle["saved_paths"]
    assert "warnings_txt" in bundle["saved_paths"]
    assert "alt_text_txt" in bundle["saved_paths"]
    assert "state_evidence_panel_png" in bundle["saved_paths"]
    assert "state_evidence_panel_svg" in bundle["saved_paths"]
    assert "state_graph_summary" in bundle["global_diagnostics"]
    assert "consensus_label_rules" in bundle["rules"]
    assert "consensus_state_map" in bundle["alt_text"]
    assert "Display" not in bundle["alt_text"]["state_evidence_panel"]
    assert "display embedding only" in bundle["alt_text"]["consensus_state_map"]
    for path in bundle["saved_paths"].values():
        assert (tmp_path / Path(path).name).exists()
        assert "A_to_B" in Path(path).name

    fig = sg.pl.local_distortion_map(
        adata,
        rep_a="X_base",
        rep_b="X_rot",
        show=False,
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    fig_worst = sg.pl.local_distortion_map(
        adata,
        aggregation="worst-case",
        show=False,
    )
    assert isinstance(fig_worst, matplotlib.figure.Figure)


def test_local_distortion_map_warns_without_per_cell_values():
    import scgeo as sg
    import matplotlib.figure

    adata = _run_full_report_stack(_make_reporting_adata(), store_per_cell=False)
    with pytest.warns(RuntimeWarning, match="Per-cell local geometry values were not stored"):
        with pytest.raises(ValueError, match="per-cell local geometry values are unavailable"):
            sg.pl.local_distortion_map(adata, show=False)
    fig = sg.pl.local_distortion_map(adata, aggregation="state", node_key="state", show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert "State-level display" in fig.axes[0].get_title()


def test_state_report_multi_k_worst_case_and_global_diagnostics():
    import scgeo as sg

    adata = _make_reporting_adata()
    bad = adata.obsm["X_base"].copy()
    bad[:, 0] *= 4.0
    adata.obsm["X_bad"] = bad
    sg.tl.local_geometry_stability(
        adata,
        reps=["X_base", "X_rot", "X_bad"],
        node_key="state",
        k_values=(5, 8),
        n_boot=0,
        max_exact_cells=200,
        seed=7,
    )
    report = sg.get.state_report(
        adata,
        node_key="state",
        robust_shift_key=None,
        representation_key=None,
        local_k=None,
        pair_aggregation="median",
    )

    moved = report.set_index("state").loc["moved"]
    assert moved["local_k_values"] == "5,8"
    assert moved["local_n_k_values"] == 2
    assert moved["local_n_valid_pairs"] == 3
    assert moved["worst_global_scale_distortion"] >= moved["median_global_scale_distortion"]
    assert np.isfinite(moved["global_scale_distortion_across_k_std"])
    assert report["state_graph_agreement"].isna().all()
    assert report.attrs["global_diagnostics"]["state_graph_summary"]["records"]
    assert report.attrs["representation_diagnostics"]["records"]

    k5 = sg.get.state_report(
        adata,
        node_key="state",
        robust_shift_key=None,
        representation_key=None,
        local_k=5,
    )
    assert k5.set_index("state").loc["moved", "local_k_values"] == "5"


def test_state_report_outer_join_long_labels_and_nonfinite_values():
    import scgeo as sg

    states = [f"very_long_state_label_{i:02d}_with_extra_text" for i in range(30)]
    obs = pd.DataFrame(
        {"state": pd.Categorical(states)},
        index=[f"c{i}" for i in range(len(states))],
    )
    adata = ad.AnnData(X=np.zeros((len(states), 1)), obs=obs)
    adata.obsm["X_umap"] = np.column_stack([np.arange(len(states)), np.zeros(len(states))])
    robust_by = {
        states[0]: {
            "n_cells0": 4,
            "n_cells1": 5,
            "n_samples0": None,
            "n_samples1": None,
            "delta_norm": np.inf,
            "normalized_delta_norm": np.nan,
            "bootstrap_magnitude_ci95": [np.nan, np.nan],
            "bootstrap_directional_resultant_length": np.nan,
            "outlier_sensitivity": {},
        }
    }
    representation = {
        "params": {
            "condition_key": "condition",
            "group0": "A",
            "group1": "B",
            "alignment_pos_thr": 0.3,
            "alignment_neg_thr": -0.3,
            "consensus_label_rules": {"min_usable_representations": 2},
        },
        "consensus_state": pd.DataFrame(
            {
                "node": [states[1]],
                "n_usable_representations": [1],
                "usable_fraction": [0.5],
                "consensus_label": ["representation_unstable"],
                "magnitude_rank_mean": [1.0],
                "magnitude_rank_std": [0.0],
                "loo_rep_magnitude_max_relative_deviation": [0.0],
                "alignment_cosine_median": [np.nan],
                "aligned_fraction": [np.nan],
                "discordant_fraction": [np.nan],
                "neutral_fraction": [np.nan],
            }
        ),
        "per_rep_state": pd.DataFrame(
            {"node": [states[1]], "magnitude_rank": [1.0], "n_cells0": [10], "n_cells1": [10]}
        ),
    }
    local = {
        "params": {"k_values": [5], "node_key": "state"},
        "state_pair_summary": pd.DataFrame(
            {
                "rep_a": ["r0"],
                "rep_b": ["r1"],
                "k": [5],
                "scope": ["state"],
                "state": [states[2]],
                "metric": ["local_distortion_median"],
                "status": ["ok"],
                "median": [np.inf],
            }
        ),
        "pair_summary": pd.DataFrame(),
        "state_graph_summary": pd.DataFrame(),
    }
    adata.uns["scgeo"] = {
        "robust_shift": {
            "params": {
                "condition_key": "condition",
                "group0": "A",
                "group1": "B",
                "bootstrap_unit": "cell",
            },
            "by": robust_by,
        },
        "representation_stability": representation,
        "local_geometry_stability": local,
    }
    report = sg.get.state_report(adata, node_key="state")
    assert report.shape[0] == 30
    assert states[0] in set(report["state"])
    assert report.set_index("state").loc[states[0], "effect_status"] == "numerical_degeneracy"
    assert report.set_index("state").loc[states[1], "stability_status"] == "representation_unstable"
    assert report.set_index("state").loc[states[2], "local_geometry_status"] == "numerical_degeneracy"

    fig, data = sg.pl.state_evidence_panel(report, max_states=5, return_data=True, show=False)
    assert data["report"].shape[0] == 5
    assert fig.axes[0].get_yticklabels()[0].get_text().startswith("very_long_state_label")


def test_state_report_and_panel_with_one_state():
    import matplotlib.figure
    import scgeo as sg

    adata = _make_reporting_adata()
    adata = adata[adata.obs["state"].astype(str) == "moved"].copy()
    _run_full_report_stack(adata)
    report = sg.get.state_report(adata, node_key="state")
    assert report["state"].tolist() == ["moved"]
    fig = sg.pl.state_evidence_panel(report, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_reporting_and_plotting_do_not_mutate_adata():
    import scgeo as sg

    adata = _run_full_report_stack(_make_reporting_adata())
    adata.obs["state"] = adata.obs["state"].astype("category")
    before = _snapshot_adata(adata)

    report = sg.get.state_report(adata, node_key="state")
    sg.pl.state_evidence_panel(report, show=False)
    sg.pl.representation_stability_heatmap(adata, show=False)
    sg.pl.consensus_state_map(adata, node_key="state", show=False)
    sg.pl.local_distortion_map(adata, aggregation="per_cell", show=False)

    _assert_adata_snapshot_equal(adata, before)


def test_lazy_matplotlib_error_message(monkeypatch):
    import scgeo.pl._perturbation_report as report_pl

    def _raise_import_error():
        raise ImportError("matplotlib is required for ScGeo plotting functions.")

    monkeypatch.setattr(report_pl, "_lazy_matplotlib", _raise_import_error)
    with pytest.raises(ImportError, match="matplotlib is required"):
        report_pl.state_evidence_panel(pd.DataFrame({"state": ["s0"]}), show=False)
