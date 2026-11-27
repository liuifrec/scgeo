"""Regression checks for plot semantics, composability, and faithful exports."""

import copy
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import Normalize

import scgeo as sg


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def report():
    table = pd.DataFrame(
        {
            "state": ["low", "high", "missing"],
            "delta_norm": [0.07, 0.8, np.nan],
            "normalized_delta_norm": [0.9, 0.2, np.nan],
            "magnitude_ci95_low": [0.12, 0.6, np.nan],
            "magnitude_ci95_high": [0.28, 1.0, np.nan],
            "n_samples0": [3, 4, np.nan],
            "n_samples1": [3, 4, np.nan],
            "bootstrap_unit": ["sample", "sample", None],
            "inference_level": ["biological_sample", "biological_sample", None],
            "representation_consensus_label": ["stable_neutral", "stable_effect", None],
            "representation_coverage_fraction": [1.0, 1.0, np.nan],
            "effect_status": ["assessed", "assessed", "numerical_degeneracy"],
            "stability_status": ["assessed", "assessed", "not_computed"],
            "local_geometry_status": [
                "not_computed",
                "insufficient_coverage",
                "not_computed",
            ],
            "dynamics_status": ["unavailable", "assessed", "not_computed"],
        }
    )
    table.attrs["provenance"] = {
        "group0": "control",
        "group1": "treated",
        "comparison_label": "treated_vs_control",
        "rep": "X_primary",
    }
    return table


@pytest.fixture
def representations():
    return pd.DataFrame(
        {
            "node": ["b", "b", "a", "a", "missing", "missing"],
            "rep": ["r2", "r1"] * 3,
            "alignment_cosine": [0.9, 0.8, -0.3, np.nan, np.nan, np.nan],
            "alignment_class": ["aligned", "aligned", "discordant", None, None, None],
            "normalized_delta_norm": [0.7, 0.8, 0.1, 0.2, np.nan, np.nan],
            "magnitude_rank": [1, 1, 2, 2, np.nan, np.nan],
        }
    )


def test_exact_interval_endpoints_and_default_metric_order(report):
    before = report.copy(deep=True)
    result = sg.pl.state_evidence_panel(report, return_result=True, show=False)
    assert result.data["report"].state.tolist() == ["high", "low", "missing"]
    intervals = result.artists["effect_intervals"].get_segments()
    np.testing.assert_allclose(intervals[1][:, 0], [0.12, 0.28])
    np.testing.assert_allclose(
        result.artists["effect_estimates"].get_offsets()[1], [0.07, 1]
    )
    assert result.metadata["effect_metric"] == "delta_norm"
    assert "X_primary" in result.axes["effect"].get_xlabel()
    pd.testing.assert_frame_equal(report, before)


def test_normalized_metric_never_reuses_raw_interval(report):
    with pytest.warns(UserWarning, match="Raw magnitude intervals were omitted"):
        result = sg.pl.state_evidence_panel(
            report,
            effect_metric="normalized_delta_norm",
            return_result=True,
            show=False,
        )
    assert "effect_intervals" not in result.artists
    assert result.data["report"].state.tolist() == ["low", "high", "missing"]
    assert result.warnings
    report["normalized_magnitude_ci95_low"] = [0.7, 0.1, np.nan]
    report["normalized_magnitude_ci95_high"] = [1.0, 0.3, np.nan]
    normalized = sg.pl.state_evidence_panel(
        report, effect_metric="normalized_delta_norm", return_result=True, show=False
    )
    np.testing.assert_allclose(
        normalized.artists["effect_intervals"].get_segments()[0][:, 0], [0.7, 1.0]
    )


def test_invalid_and_partially_missing_intervals(report):
    report.loc[0, "magnitude_ci95_low"] = 0.5
    with pytest.raises(ValueError, match="lower endpoint exceeds upper"):
        sg.pl.state_evidence_panel(report, show=False)
    report.loc[0, "magnitude_ci95_low"] = np.nan
    with pytest.warns(UserWarning, match="Incomplete or non-finite"):
        result = sg.pl.state_evidence_panel(report, return_result=True, show=False)
    assert len(result.artists["effect_intervals"].get_segments()) == 1
    assert len(result.artists["effect_estimates"].get_offsets()) == 2


def test_full_evidence_uses_supplied_axes_and_preserves_statuses(report):
    fig, axes = plt.subplots(1, 4)
    mapping = dict(zip(["effect", "samples", "stability", "availability"], axes))
    positions = [ax.get_position().bounds for ax in axes]
    figures_before = list(plt.get_fignums())
    result = sg.pl.state_evidence_panel(
        report,
        axes=mapping,
        state_order=["low", "missing", "high"],
        evidence_mode="full",
        show_sample_counts=True,
        show_inference=True,
        return_result=True,
        show=False,
    )
    assert result.fig is fig
    assert plt.get_fignums() == figures_before
    assert [ax.get_position().bounds for ax in axes] == positions
    assert result.data["report"].state.tolist() == ["low", "missing", "high"]
    availability = result.data["availability"].set_index("state")
    assert availability.loc["missing", "effect"] == "numerical_degeneracy"
    assert availability.loc["low", "dynamics"] == "unavailable"
    assert availability.loc["high", "local"] == "insufficient_coverage"
    assert "control / treated" in result.axes["samples"].get_title()
    assert any(
        "biological sample" in text.get_text() for text in result.axes["samples"].texts
    )


def test_pagination_and_explicit_order_validate(report):
    result = sg.pl.state_evidence_panel(
        report, max_states=2, page=1, return_result=True, show=False
    )
    assert result.data["report"].state.tolist() == ["missing"]
    assert result.metadata["displayed_states"] == 1
    assert result.metadata["total_states"] == 3
    assert "1 of 3 states" in result.fig._suptitle.get_text()
    with pytest.raises(ValueError, match="unknown labels"):
        sg.pl.state_evidence_panel(report, state_order=["absent"], show=False)
    with pytest.raises(ValueError, match="positive integer"):
        sg.pl.state_evidence_panel(report, max_states=0, show=False)
    with pytest.raises(ValueError, match="no states"):
        sg.pl.state_evidence_panel(report, max_states=2, page=2, show=False)


def test_heatmap_cosine_scale_missingness_and_input_order(representations):
    result = sg.pl.representation_stability_heatmap(
        representations,
        metric="alignment_cosine",
        annotate_consensus=False,
        return_result=True,
        show=False,
    )
    assert result.data["matrix"].index.tolist() == ["b", "a", "missing"]
    assert result.data["matrix"].columns.tolist() == ["r2", "r1"]
    image = result.artists["heatmap"]
    assert image.get_clim() == (-1, 1)
    assert image.norm(0) == 0.5
    assert bool(np.ma.getmaskarray(image.get_array())[1, 1])
    assert np.isnan(result.data["matrix"].loc["a", "r1"])


def test_heatmap_class_legend_is_discrete(representations):
    result = sg.pl.representation_stability_heatmap(
        representations,
        metric="alignment_class",
        annotate_consensus=False,
        return_result=True,
        show=False,
    )
    assert result.artists["heatmap"].cmap.N == 3
    assert "colorbar" not in result.axes
    assert [text.get_text() for text in result.axes["heatmap"].get_legend().texts] == [
        "discordant",
        "neutral",
        "aligned",
        "missing",
    ]
    with pytest.raises(ValueError, match="discrete scale"):
        sg.pl.representation_stability_heatmap(
            representations, metric="alignment_class", vmin=-1, show=False
        )


def test_heatmap_explicit_and_legacy_orders(representations):
    with pytest.warns(FutureWarning, match="legacy mean sorting"):
        result = sg.pl.representation_stability_heatmap(
            representations,
            cluster_states=True,
            annotate_consensus=False,
            return_result=True,
            show=False,
        )
    assert result.data["matrix"].index.tolist() == ["a", "b", "missing"]
    reordered = sg.pl.representation_stability_heatmap(
        representations,
        state_order=["missing", "b", "a"],
        rep_order=["r1", "r2"],
        annotate_consensus=False,
        return_result=True,
        show=False,
    )
    assert reordered.data["matrix"].index.tolist() == ["missing", "b", "a"]
    assert reordered.data["matrix"].columns.tolist() == ["r1", "r2"]
    clustered = sg.pl.representation_stability_heatmap(
        representations,
        order_by={"states": "cluster"},
        annotate_consensus=False,
        return_result=True,
        show=False,
    )
    assert clustered.data["matrix"].index[-1] == "missing"
    assert clustered.data["matrix"].isna().sum().sum() == 2


def test_heatmap_supplied_axes_and_norm(representations):
    fig, (ax, cax) = plt.subplots(1, 2)
    positions = [item.get_position().bounds for item in (ax, cax)]
    norm = Normalize(0, 2)
    result = sg.pl.representation_stability_heatmap(
        representations,
        norm=norm,
        annotate_consensus=False,
        axes={"heatmap": ax, "colorbar": cax},
        return_result=True,
        show=False,
    )
    assert result.fig is fig
    assert result.artists["heatmap"].norm is norm
    assert [item.get_position().bounds for item in (ax, cax)] == positions
    with pytest.raises(ValueError, match="not both"):
        sg.pl.representation_stability_heatmap(
            representations, norm=norm, vmin=0, show=False
        )


def test_required_report_failure_writes_nothing(report, tmp_path):
    adata = ad.AnnData(
        np.zeros((2, 1)), obs=pd.DataFrame({"state": ["a", "b"]}, index=["c1", "c2"])
    )
    existing = plt.figure()
    destination = tmp_path / "strict"
    with pytest.raises(ValueError, match="Required report panels"):
        sg.pl.perturbation_report(
            adata,
            node_key="state",
            report=report,
            save_dir=destination,
            strict=True,
            show=False,
        )
    assert not destination.exists()
    assert plt.fignum_exists(existing.number)
    assert plt.get_fignums() == [existing.number]


def test_report_pdf_export_includes_plotted_sources_and_metadata(report, tmp_path):
    adata = ad.AnnData(
        np.zeros((2, 1)), obs=pd.DataFrame({"state": ["a", "b"]}, index=["c1", "c2"])
    )
    bundle = sg.pl.perturbation_report(
        adata,
        node_key="state",
        report=report,
        formats=("pdf",),
        width_mm=180,
        dpi=144,
        save_dir=tmp_path,
        strict=True,
        required_panels=["state_evidence_panel"],
        panel_kwargs={"state_evidence_panel": {"max_states": 2}},
        show=False,
    )
    paths = bundle["saved_paths"]
    assert Path(paths["state_evidence_panel_pdf"]).read_bytes().startswith(b"%PDF")
    assert "state_evidence_panel_png" not in paths
    source = pd.read_csv(paths["state_evidence_panel_report_csv"])
    assert source.state.tolist() == ["high", "low"]
    assert source.set_index("state").loc["low", "magnitude_ci95_low"] == 0.12
    metadata = json.loads(Path(paths["metadata_json"]).read_text())
    assert metadata["export_config"]["width_mm"] == 180
    assert metadata["panels"]["state_evidence_panel"]["displayed_states"] == 2
    assert metadata["warnings"]


def _map_adata():
    adata = ad.AnnData(
        np.zeros((12, 1)),
        obs=pd.DataFrame(
            {"state": ["a"] * 10 + ["rare"] * 2}, index=[f"c{i}" for i in range(12)]
        ),
    )
    adata.obsm["custom"] = np.column_stack([np.arange(12), np.zeros(12)])
    adata.uns["scgeo"] = {
        "representation_stability": {
            "per_rep_state": pd.DataFrame(
                {
                    "node": ["a", "rare"],
                    "rep": ["r", "r"],
                    "normalized_delta_norm": [0.2, 0.8],
                }
            ),
            "consensus_state": pd.DataFrame(
                {
                    "node": ["a", "rare"],
                    "consensus_label": ["stable_neutral", "representation_unstable"],
                }
            ),
        },
        "local_geometry_stability": {
            "params": {"node_key": "state"},
            "per_cell": pd.DataFrame(
                {
                    "cell_index": np.arange(12),
                    "local_distortion_median": np.arange(12, dtype=float),
                    "rep_a": ["r1"] * 12,
                    "rep_b": ["r2"] * 12,
                }
            ),
        },
    }
    return adata


def test_consensus_map_sampling_retains_groups_and_input():
    adata = _map_adata()
    before = copy.deepcopy(adata.uns)
    fig, ax = plt.subplots()
    result = sg.pl.consensus_state_map(
        adata,
        node_key="state",
        basis="custom",
        max_points=3,
        random_state=2,
        required_indices=[0],
        ax=ax,
        return_result=True,
        show=False,
    )
    assert result.fig is fig
    assert result.data.displayed.sum() == 3
    assert set(result.data.loc[result.data.displayed, "state"]) == {"a", "rare"}
    assert result.data.loc[0, "displayed"]
    pd.testing.assert_frame_equal(
        adata.uns["scgeo"]["representation_stability"]["consensus_state"],
        before["scgeo"]["representation_stability"]["consensus_state"],
    )
    second = sg.pl.consensus_state_map(
        adata,
        node_key="state",
        basis="custom",
        max_points=3,
        random_state=2,
        required_indices=[0],
        return_result=True,
        show=False,
    )
    assert result.data.displayed.equals(second.data.displayed)


def test_local_distortion_sampling_preserves_full_data_scale_and_values():
    adata = _map_adata()
    result = sg.pl.local_distortion_map(
        adata,
        basis="custom",
        max_points=2,
        random_state=4,
        return_result=True,
        show=False,
    )
    assert result.data.displayed.sum() == 2
    assert result.artists["points"].get_clim() == (0, 11)
    np.testing.assert_array_equal(result.data.value, np.arange(12))
    assert result.metadata["display_only"]


def test_invalid_intervals_leave_existing_axes_untouched(report):
    report.loc[0, "magnitude_ci95_low"] = 2
    report.loc[0, "magnitude_ci95_high"] = 1
    before = list(plt.get_fignums())
    with pytest.raises(ValueError, match="lower endpoint"):
        sg.pl.state_evidence_panel(report, show=False)
    assert plt.get_fignums() == before
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("Existing title")
    children = [tuple(ax.get_children()) for ax in axes]
    with pytest.raises(ValueError, match="lower endpoint"):
        sg.pl.state_evidence_panel(
            report, axes=dict(zip(["effect", "stability"], axes)), show=False
        )
    assert [tuple(ax.get_children()) for ax in axes] == children
    assert axes[0].get_title() == "Existing title"
    assert plt.get_fignums() == [fig.number]


def test_single_interval_column_reports_unavailability(report):
    report = report.drop(columns="magnitude_ci95_high")
    with pytest.warns(UserWarning, match="endpoint columns are missing"):
        result = sg.pl.state_evidence_panel(report, return_result=True, show=False)
    assert "effect_intervals" not in result.artists
    assert len(result.artists["effect_estimates"].get_offsets()) == 2
    assert "magnitude_ci95_high" in result.warnings[0]


def test_zero_and_false_condition_labels_are_preserved(report):
    report.attrs["provenance"].update(group0=0, group1=1)
    result = sg.pl.state_evidence_panel(
        report, show_sample_counts=True, return_result=True, show=False
    )
    assert result.axes["samples"].get_title() == "Samples\n0 / 1"
    report.attrs["provenance"].update(group0=False, group1=True)
    result = sg.pl.state_evidence_panel(
        report, show_sample_counts=True, return_result=True, show=False
    )
    assert result.axes["samples"].get_title() == "Samples\nFalse / True"


@pytest.mark.parametrize(
    "metric,value",
    [
        ("alignment_cosine", 2.0),
        ("alignment_cosine", -1.1),
        ("coverage_fraction", 1.1),
        ("usable_fraction", -0.1),
    ],
)
def test_heatmap_validates_metric_domain_before_touching_axes(metric, value):
    per = pd.DataFrame({"node": ["s"], "rep": ["r"], metric: [value]})
    fig, ax = plt.subplots()
    ax.set_title("Preserved")
    children = tuple(ax.get_children())
    with pytest.raises(ValueError, match="must lie in"):
        sg.pl.representation_stability_heatmap(
            per,
            metric=metric,
            annotate_consensus=False,
            axes={"heatmap": ax},
            show=False,
        )
    assert tuple(ax.get_children()) == children
    assert plt.get_fignums() == [fig.number]
    with pytest.raises(ValueError, match="must lie in"):
        sg.pl.representation_stability_heatmap(
            per, metric=metric, annotate_consensus=False, show=False
        )
    assert plt.get_fignums() == [fig.number]


def test_heatmap_title_colorbar_consensus_and_diagnostics_do_not_overlap(
    representations,
):
    payload = {
        "per_rep_state": representations,
        "consensus_state": pd.DataFrame(
            {
                "node": ["b", "a", "missing"],
                "consensus_label": [
                    "stable_aligned",
                    "representation_unstable",
                    "insufficient_coverage",
                ],
                "usable_fraction": [1.0, 1.0, 0.0],
            }
        ),
        "representation_diagnostics": pd.DataFrame(
            {"rep": ["r2", "r1"], "neighborhood_outlier": [False, True]}
        ),
    }
    result = sg.pl.representation_stability_heatmap(
        payload,
        metric="alignment_cosine",
        title="Stored alignment · related representation views",
        return_result=True,
        show=False,
    )
    result.fig.canvas.draw()
    renderer = result.fig.canvas.get_renderer()
    bar_box = result.axes["colorbar"].yaxis.label.get_window_extent(renderer)
    assert all(
        bar_box.x0 > text.get_window_extent(renderer).x1
        for text in result.axes["consensus"].texts
    )
    title_box = result.fig._suptitle.get_window_extent(renderer)
    assert title_box.y0 > result.axes["consensus"].title.get_window_extent(renderer).y1
    assert title_box.y0 > result.axes["heatmap"].title.get_window_extent(renderer).y1
    np.testing.assert_allclose(
        result.axes["heatmap"].get_position().intervalx,
        result.axes["diagnostics"].get_position().intervalx,
    )


def test_coverage_display_label_preserves_original_status(report):
    report = report.copy()
    report["coverage_status"] = "qualified_per_mouse_before_scgeo"
    report.attrs["provenance"] = {}
    report["group0"] = "control"
    report["group1"] = "17Gy early"
    result = sg.pl.state_evidence_panel(report, show_sample_counts=True,
                                       return_result=True, show=False)
    assert "control / 17Gy early" in result.axes["samples"].get_title()
    assert any("Eligible mice verified" in item.get_text()
               for item in result.axes["coverage"].texts)
    assert set(report["coverage_status"]) == {"qualified_per_mouse_before_scgeo"}
