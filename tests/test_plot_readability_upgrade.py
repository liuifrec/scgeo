"""Readable, sample-aware plotting without changing underlying observations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.colors import Normalize

import scgeo as sg


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _consensus_data():
    obs = pd.DataFrame(
        {
            "state": ["A"] * 6 + ["B"] * 4 + ["C"] * 2,
            "sample": ["C1"] * 3 + ["C2"] * 3 + ["C1"] * 2 + ["C2"] * 2 + ["C3"] * 2,
            "score": [
                0.9,
                0.8,
                np.nan,
                0.1,
                np.inf,
                0.7,
                0.6,
                0.2,
                np.nan,
                0.95,
                0.1,
                0.3,
            ],
        },
        index=[f"synthetic_{i}" for i in range(12)],
    )
    adata = AnnData(np.zeros((12, 1)), obs=obs)
    adata.obsm["X_umap"] = np.column_stack(
        [np.arange(12) % 4, np.arange(12) // 4]
    ).astype(float)
    adata.obsm["X_umap"][9] = np.nan
    return adata


def _structure(adata, **kwargs):
    options = {
        "score_key": "score",
        "groupby": "state",
        "show": False,
        "print_legend": False,
        "colorbar": False,
        "return_result": True,
    }
    options.update(kwargs)
    return sg.pl.consensus_structure(adata, **options)


def _atlas_data(n_states=20):
    rows, points = [], []
    for state in range(n_states):
        for cell in range(4):
            condition = "control" if cell % 2 == 0 else "treated"
            rows.append(
                {
                    "state": f"S{state:02d}",
                    "condition": condition,
                    "sample": f"{condition}_{cell // 2}",
                }
            )
            points.append(
                [2 * (state % 5) + 0.03 * cell, 2 * (state // 5) + 0.03 * cell]
            )
    obs = pd.DataFrame(rows, index=[f"synthetic_{i}" for i in range(len(rows))])
    obs["state"] = pd.Categorical(obs.state, categories=sorted(obs.state.unique()))
    obs["score"] = np.linspace(-2.0, 2.0, len(obs))
    adata = AnnData(np.zeros((len(obs), 1)), obs=obs)
    adata.obsm["X_umap"] = np.array(points)
    graph = np.zeros((n_states, n_states))
    for state in range(n_states - 1):
        graph[state, state + 1] = graph[state + 1, state] = 0.5
    adata.uns["paga"] = {
        "groups": "state",
        "node_order": list(obs.state.cat.categories),
        "connectivities": graph,
    }
    return adata


def _atlas(adata, **kwargs):
    options = {
        "color": "state",
        "node_key": "state",
        "condition_key": "condition",
        "group0": "control",
        "group1": "treated",
        "panels": ("states",),
        "label_states": False,
        "legend": False,
        "show": False,
    }
    options.update(kwargs)
    return sg.pl.geometry_dynamics_atlas(adata, **options)


def test_consensus_face_outline_and_colorbar_opacities_are_independent():
    result = _structure(
        _consensus_data(), topk=3, hi_alpha=0.2, outline_alpha=0.8, colorbar=True
    )
    result.fig.canvas.draw()
    faces = result.artists["selected"]
    outlines = result.artists["selected_outlines"]
    assert faces.get_alpha() == 0.2 and outlines.get_alpha() == 0.8
    np.testing.assert_allclose(faces.get_facecolors()[:, 3], 0.2)
    np.testing.assert_allclose(outlines.get_edgecolors()[:, 3], 0.8)
    assert outlines.get_facecolors().size == 0
    assert result.artists["colorbar"].mappable is result.artists["score_mappable"]
    assert result.artists["score_mappable"].norm is faces.norm
    assert result.artists["score_mappable"].to_rgba(0.9)[3] == 1.0


def test_consensus_threshold_and_prevalence_have_full_distinct_denominators():
    result = _structure(
        _consensus_data(), selection="threshold", threshold=0.75, sample_key="sample"
    )
    cells = result.data["cells"]
    assert cells.selected.sum() == 3 and cells.highlighted.sum() == 2
    summary = result.data["summary"].set_index("group")
    a = summary.loc["A"]
    assert a.n_cells == 6 and a.n_scored == 4 and a.n_unscored == 2
    assert a["count"] == 2 and a.fraction == pytest.approx(2 / 3)
    assert a.score_coverage == pytest.approx(4 / 6)
    assert a.selected_fraction_all_cells == pytest.approx(2 / 6)
    assert a.selected_fraction_scored_cells == pytest.approx(2 / 4)
    samples = result.data["samples"].set_index(["group", "sample"])
    assert len(samples) == 9
    assert samples.loc[("A", "C1"), "fraction"] == 1.0
    absent = samples.loc[("A", "C3")]
    assert absent["count"] == absent.n_cells == 0 and absent.sample_n_cells == 2
    assert pd.isna(absent.selected_fraction_all_cells)
    assert absent.status == "no_cells_in_group"
    assert result.data["summary"]["count"].sum() == 3


def test_consensus_sampling_retains_selected_cells_and_group_sample_strata():
    adata = _consensus_data()
    options = {"selection": "threshold", "threshold": 0.75, "sample_key": "sample"}
    full = _structure(adata, **options)
    small = _structure(adata, max_points=6, random_state=9, **options)
    repeated = _structure(adata, max_points=6, random_state=9, **options)
    cells = small.data["cells"]
    assert cells.background_displayed.sum() == 6
    assert cells.loc[cells.highlighted, "background_displayed"].all()
    visible = cells.loc[cells.background_displayed]
    expected = set(
        map(
            tuple,
            cells.loc[cells.finite_coordinates, ["group_label", "sample_label"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    actual = set(
        map(
            tuple, visible[["group_label", "sample_label"]].drop_duplicates().to_numpy()
        )
    )
    assert actual == expected
    pd.testing.assert_frame_equal(small.data["summary"], full.data["summary"])
    pd.testing.assert_frame_equal(small.data["samples"], full.data["samples"])
    pd.testing.assert_frame_equal(cells, repeated.data["cells"])
    assert small.metadata["norm"] == full.metadata["norm"]


def test_consensus_too_small_budget_raises_before_changing_caller_axes():
    _, ax = plt.subplots()
    children = list(ax.get_children())
    with pytest.raises(
        ValueError, match="2 selected visible cells.*5 observed display strata"
    ):
        _structure(
            _consensus_data(),
            selection="threshold",
            threshold=0.75,
            sample_key="sample",
            max_points=5,
            ax=ax,
        )
    assert list(ax.get_children()) == children


def test_consensus_empty_threshold_and_legacy_topk_return():
    empty = _structure(_consensus_data(), selection="threshold", threshold=2.0)
    assert empty.metadata["n_selected"] == 0
    assert empty.data["summary"]["count"].eq(0).all()
    legacy = _structure(_consensus_data(), topk=3, return_result=False)
    assert len(legacy) == 4 and legacy[2].topk == 3 and legacy[2].n_selected == 3
    assert legacy[2].selection == "topk"


def test_consensus_missing_samples_are_explicit_and_never_dropped():
    adata = _consensus_data()
    adata.obs.loc[adata.obs_names[0], "sample"] = None
    result = _structure(adata, topk=3, sample_key="sample")
    samples = result.data["samples"]
    missing = samples.loc[samples.sample_is_missing]
    assert len(missing) == 3 and missing.n_cells.sum() == 1
    assert missing["count"].sum() == 1
    assert samples.n_cells.sum() == adata.n_obs
    assert result.data["cells"].sample_is_missing.sum() == 1


def test_consensus_rasterization_policy_preserves_source_values():
    adata = _consensus_data()
    raster = _structure(adata, topk=3, rasterized=True)
    vector = _structure(adata, topk=3, rasterized=False)
    assert raster.artists["selected"].get_rasterized()
    assert not vector.artists["selected"].get_rasterized()
    assert all(
        artist.get_rasterized() for artist in raster.artists["background"].values()
    )
    pd.testing.assert_frame_equal(raster.data["summary"], vector.data["summary"])


def test_consensus_auto_rasterization_uses_display_count():
    n = 2001
    adata = AnnData(
        np.zeros((n, 1)),
        obs=pd.DataFrame(
            {"state": ["A"] * n, "score": np.arange(n, dtype=float)},
            index=[f"synthetic_{i}" for i in range(n)],
        ),
    )
    adata.obsm["X_umap"] = np.column_stack([np.arange(n) % 41, np.arange(n) // 41])
    dense = _structure(adata, topk=1)
    small = _structure(adata, topk=1, max_points=100)
    assert dense.artists["selected"].get_rasterized()
    assert not small.artists["selected"].get_rasterized()
    pd.testing.assert_frame_equal(dense.data["summary"], small.data["summary"])
    np.testing.assert_allclose(
        dense.axes["embedding"].get_xlim(), small.axes["embedding"].get_xlim()
    )
    np.testing.assert_allclose(
        dense.axes["embedding"].get_ylim(), small.axes["embedding"].get_ylim()
    )


@pytest.mark.parametrize(
    "options",
    [
        {"selection": "unknown"},
        {"selection": "threshold"},
        {"selection": "threshold", "threshold": np.inf},
        {"threshold": 0.5},
        {"rasterized": "unknown"},
        {"hi_alpha": 1.1},
        {"outline_alpha": -0.1},
        {"norm": Normalize(0, 1), "vmin": 0},
        {"vmin": 2, "vmax": 1},
    ],
)
def test_consensus_invalid_new_fields_fail_before_artist_creation(options):
    _, ax = plt.subplots()
    children = list(ax.get_children())
    with pytest.raises((TypeError, ValueError)):
        _structure(_consensus_data(), ax=ax, **options)
    assert list(ax.get_children()) == children


def test_atlas_shared_norm_uses_all_scores_before_display_sampling():
    adata = _atlas_data()
    norm = Normalize(-3.0, 3.0)
    result = _atlas(adata, color="score", cmap="plasma", norm=norm, max_points=20)
    assert result.artists["states_points"].norm is norm
    assert result.artists["states_points"].cmap.name == "plasma"
    assert result.metadata["panel_provenance"]["states"]["numeric_norm"] == {
        "type": "Normalize",
        "vmin": -3.0,
        "vmax": 3.0,
    }
    auto = _atlas(adata, color="score", max_points=20)
    assert auto.artists["states_points"].norm.vmin == -2.0
    assert auto.artists["states_points"].norm.vmax == 2.0
    assert len(auto.data["cells"]) == 80 and auto.data["cells"].displayed.sum() == 20


def test_atlas_label_budget_is_deterministic_and_highlights_survive_overflow():
    adata = _atlas_data()
    result = _atlas(
        adata, label_states=True, max_labels=2, highlight_states=["S17", "S18", "S19"]
    )
    labels = result.data["labels"].set_index("state")
    assert set(labels.loc[labels.displayed].index) == {"S17", "S18", "S19"}
    assert result.metadata["highlight_label_overflow"] == 1
    assert labels.loc["S00", "reason"] == "label_budget"
    assert set(result.artists["state_labels"]) == {"S17", "S18", "S19"}
    ordinary = _atlas(adata, label_states=True, max_labels=2, highlight_states=["S19"])
    assert set(ordinary.artists["state_labels"]) == {"S00", "S19"}


def test_atlas_manual_label_offsets_change_placement_not_anchors():
    adata = _atlas_data(3)
    base = _atlas(adata, label_states=True)
    moved = _atlas(adata, label_states=True, label_offsets={"S00": (35, -22)})
    row = moved.data["labels"].set_index("state").loc["S00"]
    assert row.offset_x_points == 35 and row.offset_y_points == -22
    assert row.placement == "manual"
    np.testing.assert_allclose(
        base.data["labels"][["anchor_x", "anchor_y"]],
        moved.data["labels"][["anchor_x", "anchor_y"]],
    )


def test_atlas_long_legend_has_explicit_notice_and_full_export_rows():
    result = _atlas(
        _atlas_data(), legend="auto", legend_max_items=5, highlight_states=["S19"]
    )
    legend = result.data["legend"].set_index("category")
    assert len(legend) == 20 and legend.displayed.sum() == 5
    assert legend.loc["S19", "displayed"]
    assert (
        "15 more categories" in result.artists["states_legend"].get_title().get_text()
    )
    full = _atlas(_atlas_data(), legend=True, legend_max_items=5)
    assert full.data["legend"].displayed.all()


def test_atlas_supplied_legend_and_sample_coverage_preserve_parent_geometry():
    adata = _atlas_data(3)
    adata.obs.loc[adata.obs_names[0], "sample"] = None
    adata.obsm["X_umap"][1] = np.nan
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Caller title")
    bounds = [ax.get_position(original=True).bounds for ax in axs]
    result = _atlas(
        adata,
        sample_key="sample",
        legend="auto",
        axes={"states": axs[0], "legend": axs[1]},
        layout="adaptive",
    )
    assert result.artists["states_legend"].axes is axs[1]
    text = result.artists["sample_coverage"].get_text()
    assert "control n=2" in text and "treated n=2" in text
    assert "missing sample IDs: 1 cells" in text
    coverage = result.data["sample_coverage"].set_index("condition")
    assert coverage.n_cells.sum() == 12
    assert coverage.n_finite_coordinates.sum() == 11
    assert coverage.n_missing_sample_cells.sum() == 1
    assert fig._suptitle.get_text() == "Caller title"
    np.testing.assert_array_equal(fig.get_size_inches(), [9, 4])
    np.testing.assert_allclose(
        [ax.get_position(original=True).bounds for ax in axs], bounds
    )
    assert result.metadata["resolved_layout"] == "supplied_axes"


def test_atlas_adaptive_owned_layout_preserves_evidence_and_spatial_limits():
    adata = _atlas_data(3)
    evidence = pd.DataFrame(
        {"state": ["S00", "S01", "S02"], "delta_norm": [0.2, 0.5, 0.3]}
    )
    options = {
        "report": evidence,
        "panels": ("states", "paga", "evidence"),
        "paga_options": {"min_cells": 1, "label": False},
    }
    grid = _atlas(adata, **options)
    adaptive = _atlas(adata, layout="adaptive", **options)
    assert grid.metadata["resolved_layout"] == "grid"
    assert adaptive.metadata["resolved_layout"] == "adaptive_evidence_strip"
    assert adaptive.fig.get_size_inches()[1] < grid.fig.get_size_inches()[1]
    pd.testing.assert_frame_equal(
        grid.data["evidence_all"], adaptive.data["evidence_all"]
    )
    np.testing.assert_allclose(
        adaptive.axes["states"].get_xlim(), adaptive.axes["paga"].get_xlim()
    )
    np.testing.assert_allclose(
        adaptive.axes["states"].get_ylim(), adaptive.axes["paga"].get_ylim()
    )


def test_atlas_owned_evidence_legend_does_not_cover_estimates():
    evidence = pd.DataFrame(
        {
            "state": ["S00", "S01", "S02"],
            "delta_norm": [0.2, 0.5, 0.3],
            "effect_status": ["assessed", "assessed", "limited"],
        }
    )
    result = _atlas(_atlas_data(3), report=evidence, panels=("states", "evidence"))
    result.fig.canvas.draw()
    legend = result.artists["evidence_legend"]
    renderer = result.fig.canvas.get_renderer()
    assert (
        legend.get_window_extent(renderer).y1
        < result.axes["evidence"].get_window_extent(renderer).y0
    )


@pytest.mark.parametrize("style,dpi", [("paper", 110), ("presentation", 144)])
def test_atlas_adaptive_legend_clears_xlabel_and_sample_footer(style, dpi):
    evidence = pd.DataFrame(
        {
            "state": ["S00", "S01", "S02"],
            "delta_norm": [0.2, 0.5, 0.3],
            "effect_status": ["assessed", "assessed", "limited"],
        }
    )
    result = _atlas(
        _atlas_data(3),
        report=evidence,
        panels=("states", "paga", "evidence"),
        paga_options={"min_cells": 1, "label": False},
        sample_key="sample",
        layout="adaptive",
        style=style,
    )
    result.fig.set_dpi(dpi)
    result.fig.canvas.draw()
    renderer = result.fig.canvas.get_renderer()
    legend = result.artists["evidence_legend"].get_window_extent(renderer)
    evidence_axis = result.axes["evidence"]
    xlabel = evidence_axis.xaxis.label.get_window_extent(renderer)
    footer = result.artists["sample_coverage"].get_window_extent(renderer)
    gap = 2 * dpi / 72.0
    assert legend.y1 < evidence_axis.get_window_extent(renderer).y0
    assert legend.y1 + gap < xlabel.y0
    assert footer.y1 + gap < legend.y0


@pytest.mark.parametrize(
    "options",
    [
        {"color": "score", "norm": Normalize(0, 1), "vmin": 0},
        {"color": "score", "vmin": 2, "vmax": 1},
        {"color": "score", "cmap": "invalid"},
        {"norm": Normalize(0, 1)},
        {"max_labels": -1},
        {"label_mode": "invalid"},
        {"label_offsets": {"unknown": (0, 0)}},
        {"label_offsets": {"S00": (np.nan, 0)}},
        {"legend": "invalid"},
        {"legend_max_items": 0},
        {"legend_loc": "invalid"},
        {"layout": "invalid"},
        {"show_sample_coverage": "yes"},
    ],
)
def test_atlas_invalid_new_fields_fail_before_artist_creation(options):
    _, ax = plt.subplots()
    children = list(ax.get_children())
    with pytest.raises((TypeError, ValueError)):
        _atlas(_atlas_data(3), axes={"states": ax}, show_colorbar=False, **options)
    assert list(ax.get_children()) == children


def test_new_prevalence_and_label_sources_are_serializable(tmp_path):
    consensus = _structure(
        _consensus_data(), selection="threshold", threshold=0.75, sample_key="sample"
    )
    output = consensus.save(tmp_path / "consensus", formats=("svg",))
    manifest = json.loads(Path(output["metadata"]).read_text())
    assert manifest["tables"]["samples"]["rows"] == 9
    assert manifest["plot"]["selection"] == "threshold"
    atlas = _atlas(
        _atlas_data(3), sample_key="sample", label_states=True, legend="auto"
    )
    output = atlas.save(tmp_path / "atlas", formats=("svg",))
    manifest = json.loads(Path(output["metadata"]).read_text())
    assert {"labels", "legend", "sample_coverage"} <= set(manifest["tables"])
    assert manifest["plot"]["max_labels"] == 14
