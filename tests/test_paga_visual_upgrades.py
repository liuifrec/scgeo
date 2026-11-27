"""Scientific and composability contracts for PAGA display overlays.

The fixtures deliberately include an unrelated condition, a state absent from
each side of the contrast, and unequal edge weights. These distinguish graph
identity and contrast errors from harmless changes to display styling.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.colors import to_rgba
from matplotlib.legend import Legend
from scipy import sparse

import scgeo as sg


@pytest.fixture
def paga_data():
    rows, coordinates, velocities = [], [], []
    bases = {"A": (0.0, 0.0), "B": (4.0, 1.0), "C": (2.0, 4.0), "D": (6.0, 4.0)}
    shifts = {"A": (1.0, 0.4), "B": (-0.5, 1.2), "C": (0.3, 0.6), "D": (0.5, -0.5)}
    counts = {"A": (3, 2, 2), "B": (2, 4, 2), "C": (3, 0, 1), "D": (0, 3, 1)}
    velocity = {"A": (0.2, 0.1), "B": (-0.1, 0.3), "C": (0.15, 0.0), "D": (0.0, 0.2)}
    for state in bases:
        for condition, count in zip(("control", "treated", "unrelated"), counts[state]):
            for replicate in range(count):
                rows.append({"state": state, "condition": condition})
                point = np.asarray(bases[state]) + np.array(
                    [0.05 * replicate, -0.025 * replicate]
                )
                if condition == "treated":
                    point = point + shifts[state]
                elif condition == "unrelated":
                    point = point + [8.0, 8.0]
                coordinates.append(point)
                velocities.append(
                    [100.0, 200.0] if condition == "unrelated" else velocity[state]
                )
    obs = pd.DataFrame(rows, index=[f"cell_{i:03d}" for i in range(len(rows))])
    obs["state"] = pd.Categorical(obs["state"], categories=list(bases), ordered=True)
    obs["condition"] = pd.Categorical(
        obs["condition"], categories=["control", "treated", "unrelated"]
    )
    adata = AnnData(np.zeros((len(obs), 2)), obs=obs)
    adata.obsm["X_umap"] = np.asarray(coordinates)
    adata.obsm["velocity_umap"] = np.asarray(velocities)
    adata.obsm["X_pca"] = np.column_stack([coordinates, np.arange(len(obs))])
    adata.obsm["velocity_pca"] = np.column_stack([velocities, np.ones(len(obs))])
    adata.uns["paga"] = {
        "groups": "state",
        "connectivities": sparse.csr_matrix(
            [
                [0.0, 0.9, 0.03, 0.2],
                [0.9, 0.0, 0.6, 0.4],
                [0.03, 0.6, 0.0, 0.1],
                [0.2, 0.4, 0.1, 0.0],
            ]
        ),
    }
    table = pd.DataFrame(
        {
            "node": ["A", "B", "C", "D"],
            "rep": ["X_pca"] * 4,
            "alignment_cosine": [0.8, -0.6, np.nan, np.nan],
            "usable": [True, True, False, False],
        }
    )
    adata.uns["scgeo"] = {
        "representation_stability": {
            "per_rep_state": table,
            "params": {
                "node_key": "state",
                "condition_key": "condition",
                "group0": "control",
                "group1": "treated",
            },
        }
    }
    yield adata
    plt.close("all")


def _plot(adata, **kwargs):
    options = {
        "node_key": "state",
        "condition_key": "condition",
        "group0": "control",
        "group1": "treated",
        "min_cells": 1,
        "show": False,
    }
    options.update(kwargs)
    return sg.pl.paga_shift_map(adata, **options)


def _assert_axes_untouched(ax, before):
    assert list(ax.get_children()) == before["children"]
    assert ax.get_xlim() == before["xlim"]
    assert ax.get_ylim() == before["ylim"]
    assert ax.get_title() == before["title"]


def _axes_snapshot(ax):
    return {
        "children": list(ax.get_children()),
        "xlim": ax.get_xlim(),
        "ylim": ax.get_ylim(),
        "title": ax.get_title(),
    }


def test_legacy_returns_and_supplied_limits_survive(paga_data):
    fig, ax = _plot(paga_data)
    assert ax.figure is fig
    assert float(ax.get_aspect()) == pytest.approx(1.0)
    supplied_fig, supplied_ax = plt.subplots()
    supplied_ax.set(xlim=(-2.0, 20.0), ylim=(-3.0, 15.0))
    output = _plot(paga_data, ax=supplied_ax)
    assert output is supplied_ax
    assert supplied_ax.get_xlim() == (-2.0, 20.0)
    assert supplied_ax.get_ylim() == (-3.0, 15.0)
    assert float(supplied_ax.get_aspect()) == pytest.approx(1.0)
    fig_data, ax_data, data = _plot(paga_data, return_data=True)
    assert ax_data.figure is fig_data
    assert isinstance(data["centroids"], pd.DataFrame)
    assert isinstance(data["edges"], list)
    assert supplied_fig is supplied_ax.figure


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(3),
        np.ones((4, 3)),
        [[0, 0.8, 0, 0], [0.2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, np.nan, 0, 0], [np.nan, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, np.inf, 0, 0], [np.inf, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, -0.1, 0, 0], [-0.1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ],
    ids=["wrong_node_count", "nonsquare", "asymmetric", "nan", "infinite", "negative"],
)
def test_invalid_connectivity_fails_before_axes_mutation(paga_data, matrix):
    paga_data.uns["paga"]["connectivities"] = sparse.csr_matrix(matrix)
    _, ax = plt.subplots()
    ax.set(title="Untouched", xlim=(-1, 2), ylim=(-3, 4))
    before = _axes_snapshot(ax)
    with pytest.raises(ValueError):
        _plot(paga_data, ax=ax)
    _assert_axes_untouched(ax, before)


@pytest.mark.parametrize("velocity_basis", ["pca", "velocity_pca"])
def test_wrong_projected_velocity_basis_rejected_before_drawing(
    paga_data, velocity_basis
):
    _, ax = plt.subplots()
    before = _axes_snapshot(ax)
    with pytest.raises(ValueError, match="(?i)(velocity|basis|embedding)"):
        _plot(paga_data, ax=ax, show_velocity=True, velocity_basis=velocity_basis)
    _assert_axes_untouched(ax, before)


def test_projection_and_composition_use_selected_conditions(paga_data):
    _, _, data = _plot(
        paga_data, show_velocity=True, pie_key="condition", return_data=True
    )
    velocity = data["velocity"].set_index("state")
    np.testing.assert_allclose(
        velocity.loc["A", ["vx", "vy"]].to_numpy(dtype=float), [0.2, 0.1]
    )
    np.testing.assert_allclose(
        velocity.loc["B", ["vx", "vy"]].to_numpy(dtype=float), [-0.1, 0.3]
    )
    composition = data["composition"].set_index("node")
    np.testing.assert_allclose(
        composition.loc["A", ["control", "treated"]].to_numpy(dtype=float),
        [3 / 5, 2 / 5],
    )
    if "unrelated" in composition:
        assert composition["unrelated"].eq(0).all()
    centroids = data["centroids"].set_index("state")
    assert centroids.loc["A", "n0"] == 3
    assert centroids.loc["A", "n1"] == 2
    # Unequal per-state counts must not become a denominator over all conditions.
    assert centroids.loc["A", "frac0"] == pytest.approx(3 / 8)
    assert centroids.loc["A", "frac1"] == pytest.approx(2 / 9)


def test_explicit_graph_order_keeps_edge_identity_after_category_reordering(paga_data):
    original = _plot(paga_data, return_result=True, connectivity_threshold=0.15)
    paga_data.obs["state"] = paga_data.obs["state"].cat.reorder_categories(
        ["D", "B", "A", "C"]
    )
    reordered = _plot(
        paga_data,
        graph_node_order=["A", "B", "C", "D"],
        return_result=True,
        connectivity_threshold=0.15,
    )
    columns = ["source", "target", "weight"]
    pd.testing.assert_frame_equal(
        original.data["edge_table"][columns]
        .sort_values(columns[:2])
        .reset_index(drop=True),
        reordered.data["edge_table"][columns]
        .sort_values(columns[:2])
        .reset_index(drop=True),
    )
    edge_weights = reordered.data["edge_table"].set_index(["source", "target"])[
        "weight"
    ]
    assert edge_weights.loc[("A", "B")] == pytest.approx(0.9)
    assert edge_weights.loc[("A", "C")] == pytest.approx(0.03)


@pytest.mark.parametrize(
    "order", [["A", "A", "C", "D"], ["A", "B", "C"], ["A", "B", "C", "unknown"]]
)
def test_graph_order_requires_unique_ids_and_includes_observed_states(paga_data, order):
    _, ax = plt.subplots()
    before = _axes_snapshot(ax)
    with pytest.raises(ValueError, match="(?i)(order|categor|node)"):
        _plot(paga_data, graph_node_order=order, ax=ax)
    _assert_axes_untouched(ax, before)


def test_paga_groups_must_match_requested_state_annotation(paga_data):
    paga_data.uns["paga"]["groups"] = "different_clustering"
    with pytest.raises(ValueError, match="(?i)(groups|node_key|clustering)"):
        _plot(paga_data)


def test_explicit_graph_order_preserves_absent_nodes_and_edges_after_subsetting(
    paga_data,
):
    subset = paga_data[paga_data.obs.state.ne("D")].copy()
    assert "D" not in subset.obs.state.cat.categories
    result = _plot(subset, graph_node_order=["A", "B", "C", "D"], return_result=True)
    edges = result.data["edge_table"]
    omitted = edges.loc[edges.source.eq("D") | edges.target.eq("D")]
    assert len(omitted) == 3
    assert not omitted.displayed.any()
    assert omitted.reason.str.len().gt(0).all()
    assert set(result.data["display_cells"]["cell"]) == set(subset.obs_names)


def test_string_coercion_cannot_merge_distinct_graph_node_ids(paga_data):
    paga_data.obs["state"] = paga_data.obs["state"].cat.rename_categories(
        [1, "1", "C", "D"]
    )
    with pytest.raises(ValueError, match="(?i)(unique|duplicate|ambiguous|distinct)"):
        _plot(paga_data)


def test_stored_alignment_is_invariant_to_display_coordinates(paga_data):
    original = copy.deepcopy(paga_data.uns["scgeo"]["representation_stability"])
    options = {
        "node_color_mode": "alignment",
        "alignment_source": "stored",
        "analysis_rep": "X_pca",
        "return_result": True,
    }
    first = _plot(paga_data, **options)
    paga_data.obsm["X_umap"] = paga_data.obsm["X_umap"] @ np.array(
        [[2.0, -1.0], [0.5, 3.0]]
    ) + [11.0, -5.0]
    second = _plot(paga_data, **options)
    pd.testing.assert_frame_equal(first.data["alignment"], second.data["alignment"])
    np.testing.assert_allclose(
        first.data["alignment"]["alignment_cosine"],
        [0.8, -0.6, np.nan, np.nan],
        equal_nan=True,
    )
    assert not first.data["vectors"].equals(second.data["vectors"])
    for state in ("A", "B"):
        np.testing.assert_allclose(
            first.artists["reference_nodes"][state].get_edgecolors(),
            second.artists["reference_nodes"][state].get_edgecolors(),
        )
    assert not np.allclose(
        first.artists["reference_nodes"]["A"].get_edgecolors(),
        first.artists["reference_nodes"]["B"].get_edgecolors(),
    )
    pd.testing.assert_frame_equal(
        original["per_rep_state"],
        paga_data.uns["scgeo"]["representation_stability"]["per_rep_state"],
    )
    assert (
        original["params"]
        == paga_data.uns["scgeo"]["representation_stability"]["params"]
    )


def test_multiple_stored_representations_require_explicit_selection(paga_data):
    payload = paga_data.uns["scgeo"]["representation_stability"]
    extra = payload["per_rep_state"].assign(rep="X_latent", alignment_cosine=0.25)
    payload["per_rep_state"] = pd.concat(
        [payload["per_rep_state"], extra], ignore_index=True
    )
    options = {"node_color_mode": "alignment", "alignment_source": "stored"}
    with pytest.raises(ValueError, match="analysis_rep"):
        _plot(paga_data, **options)
    with pytest.raises(ValueError, match="analysis_rep"):
        _plot(paga_data, analysis_rep="missing_rep", **options)
    result = _plot(paga_data, analysis_rep="X_latent", return_result=True, **options)
    assert result.data["alignment"].rep.eq("X_latent").all()
    assert result.data["alignment"].alignment_cosine.eq(0.25).all()


@pytest.mark.parametrize(
    "field,value",
    [
        ("group0", "treated"),
        ("group1", "unrelated"),
        ("node_key", "other"),
        ("condition_key", "batch"),
    ],
)
def test_stored_alignment_contrast_mismatch_rejected_before_drawing(
    paga_data, field, value
):
    paga_data.uns["scgeo"]["representation_stability"]["params"][field] = value
    _, ax = plt.subplots()
    before = _axes_snapshot(ax)
    with pytest.raises(ValueError, match="does not match"):
        _plot(
            paga_data,
            ax=ax,
            node_color_mode="alignment",
            alignment_source="stored",
            analysis_rep="X_pca",
        )
    _assert_axes_untouched(ax, before)


@pytest.mark.parametrize("bad_values", [[0.1, 1.2], [0.1, np.inf]])
def test_provided_alignment_rejects_values_outside_cosine_domain(paga_data, bad_values):
    table = pd.DataFrame({"state": ["A", "B"], "alignment_cosine": bad_values})
    with pytest.raises(ValueError, match="(?i)(alignment|cosine)"):
        _plot(paga_data, node_color_mode="alignment", alignment_df=table)


def test_provided_alignment_requires_unique_states(paga_data):
    table = pd.DataFrame({"state": ["A", "A"], "alignment_cosine": [0.1, -0.7]})
    with pytest.raises(ValueError, match="(?i)(unique|duplicate|one row)"):
        _plot(paga_data, node_color_mode="alignment", alignment_df=table)


def test_display_sampling_preserves_population_summaries(paga_data):
    options = {"return_result": True, "pie_key": "condition", "show_velocity": True}
    complete = _plot(paga_data, **options)
    sampled = _plot(
        paga_data, max_points=16, random_state=31, required_indices=[0, 4], **options
    )
    repeat = _plot(
        paga_data, max_points=16, random_state=31, required_indices=[0, 4], **options
    )
    for key in ("centroids", "velocity", "composition", "edge_table", "vectors"):
        pd.testing.assert_frame_equal(complete.data[key], sampled.data[key])
    cells = sampled.data["display_cells"]
    pd.testing.assert_frame_equal(cells, repeat.data["display_cells"])
    assert len(cells) == paga_data.n_obs
    assert cells["cell"].tolist() == paga_data.obs_names.tolist()
    assert cells["obs_position"].tolist() == list(range(paga_data.n_obs))
    assert int(cells["displayed"].sum()) == 16
    assert cells.loc[cells.obs_position.isin([0, 4]), "displayed"].all()
    assert set(cells.loc[cells.displayed, "state"]) == {"A", "B", "C", "D"}
    np.testing.assert_allclose(cells[["x", "y"]], paga_data.obsm["X_umap"])


def test_export_retains_filtered_edges_all_cells_and_plot_provenance(
    paga_data, tmp_path
):
    # Context-only cells remain in the source even if they cannot be displayed.
    assert paga_data.obs.iloc[5]["condition"] == "unrelated"
    paga_data.obsm["X_umap"][5] = [np.nan, 0.0]
    result = _plot(
        paga_data, max_points=16, connectivity_threshold=0.5, return_result=True
    )
    assert isinstance(result, sg.pl.PlotResult)
    edges = result.data["edge_table"]
    weak = edges.loc[edges.source.eq("A") & edges.target.eq("C")].iloc[0]
    assert weak.weight == pytest.approx(0.03)
    assert not weak.displayed
    assert isinstance(weak.reason, str) and weak.reason
    cells = result.data["display_cells"]
    assert len(cells) == paga_data.n_obs
    assert not cells.loc[cells.obs_position.eq(5), "displayed"].iloc[0]
    assert np.isnan(cells.loc[cells.obs_position.eq(5), "x"].iloc[0])
    output = result.save(tmp_path / "paga", formats=("svg", "png"), dpi=60)
    manifest = json.loads(Path(output["metadata"]).read_text())
    for key in ("edge_table", "display_cells", "centroids", "vectors", "labels"):
        source_path = Path(output["tables"][key])
        reloaded = pd.read_csv(source_path)
        pd.testing.assert_frame_equal(reloaded, result.data[key], check_dtype=False)
        assert (
            manifest["tables"][key]["sha256"]
            == hashlib.sha256(source_path.read_bytes()).hexdigest()
        )
    assert set(output["figures"]) == {"svg", "png"}
    assert manifest["plot"]
    json.dumps(result.metadata, allow_nan=False)


def test_missing_condition_keeps_actual_anchor_without_shift_arrow(paga_data):
    result = _plot(paga_data, return_result=True)
    centers = result.data["centroids"].set_index("state")
    vectors = result.data["vectors"]
    shifts = vectors.loc[vectors.kind.eq("shift")].set_index("state")
    assert not bool(centers.loc["C", "present1"])
    assert not bool(centers.loc["D", "present0"])
    assert centers.loc["C", "n1"] == 0
    assert centers.loc["D", "n0"] == 0
    for state in ("C", "D"):
        assert not shifts.loc[state, "displayed"]
        assert shifts.loc[state, "reason"]
    labels = result.data["labels"].set_index("state")
    np.testing.assert_allclose(
        labels.loc["C", ["anchor_x", "anchor_y"]].to_numpy(dtype=float),
        centers.loc["C", ["x0", "y0"]].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        labels.loc["D", ["anchor_x", "anchor_y"]].to_numpy(dtype=float),
        centers.loc["D", ["x1", "y1"]].to_numpy(dtype=float),
    )


def test_recorded_arrow_endpoints_follow_scale_and_are_in_view(paga_data):
    result = _plot(
        paga_data,
        show_velocity=True,
        arrow_scale=25.0,
        velocity_scale=90.0,
        return_result=True,
    )
    centers = result.data["centroids"].set_index("state")
    vectors = result.data["vectors"]
    for _, vector in vectors.loc[
        vectors.kind.eq("shift") & vectors.displayed
    ].iterrows():
        center = centers.loc[vector.state]
        np.testing.assert_allclose(
            [vector.target_x, vector.target_y],
            [center.x0 + 25 * center.dx, center.y0 + 25 * center.dy],
        )
        np.testing.assert_allclose(
            [vector.comparison_x, vector.comparison_y], [center.x1, center.y1]
        )
        np.testing.assert_allclose(
            result.artists["comparison_nodes"][vector.state].get_offsets(),
            [[center.x1, center.y1]],
        )
    # The display scaling is deliberately large enough to leave the cell cloud.
    axis = next(iter(result.axes.values()))
    xlim, ylim = sorted(axis.get_xlim()), sorted(axis.get_ylim())
    drawn = vectors.loc[vectors.displayed]
    assert drawn.target_x.between(*xlim).all()
    assert drawn.target_y.between(*ylim).all()
    assert not np.allclose(
        drawn.loc[drawn.kind.eq("shift"), ["target_x", "target_y"]],
        drawn.loc[drawn.kind.eq("shift"), ["comparison_x", "comparison_y"]],
    )


def test_label_offsets_preserve_centroids_and_arrow_geometry(paga_data):
    baseline = _plot(paga_data, return_result=True)
    offsets = {"A": (31.0, -17.0), "B": (-23.0, 19.0)}
    changed = _plot(paga_data, label_offsets=offsets, return_result=True)
    pd.testing.assert_frame_equal(baseline.data["centroids"], changed.data["centroids"])
    pd.testing.assert_frame_equal(baseline.data["vectors"], changed.data["vectors"])
    base_labels = baseline.data["labels"].set_index("state")
    labels = changed.data["labels"].set_index("state")
    pd.testing.assert_frame_equal(
        base_labels[["anchor_x", "anchor_y"]], labels[["anchor_x", "anchor_y"]]
    )
    for state, requested in offsets.items():
        np.testing.assert_allclose(
            labels.loc[state, ["offset_x_points", "offset_y_points"]].to_numpy(
                dtype=float
            ),
            requested,
        )


def test_explicit_limits_and_subfigure_export_contract(paga_data, tmp_path):
    fig = plt.figure(figsize=(8.0, 4.0))
    subfigures = fig.subfigures(1, 2)
    ax = subfigures[0].subplots()
    sibling = subfigures[1].subplots()
    sibling.set_title("Independent panel")
    sibling.plot([0, 1], [1, 0])
    sibling_before = _axes_snapshot(sibling)
    size_before = fig.get_size_inches().copy()
    result = _plot(
        paga_data,
        ax=ax,
        xlim=(-2.0, 18.0),
        ylim=(-4.0, 16.0),
        node_color_mode="alignment",
        alignment_source="stored",
        analysis_rep="X_pca",
        show_colorbar=False,
        return_result=True,
    )
    assert result.fig is fig
    assert ax in result.axes.values()
    assert ax.get_xlim() == (-2.0, 18.0)
    assert ax.get_ylim() == (-4.0, 16.0)
    np.testing.assert_array_equal(fig.get_size_inches(), size_before)
    _assert_axes_untouched(sibling, sibling_before)
    assert result.metadata["export_scope"] == "parent_figure"
    assert "sibling" in result.metadata["source_scope"]
    result.save(tmp_path / "composed", formats="png", dpi=40)
    np.testing.assert_array_equal(fig.get_size_inches(), size_before)
    _assert_axes_untouched(sibling, sibling_before)


def test_pie_metric_rings_keep_numeric_evidence_visible_and_fixed_size(paga_data):
    result = _plot(
        paga_data,
        pie_key="condition",
        node_color_mode="alignment",
        alignment_source="stored",
        analysis_rep="X_pca",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        node_scale_by_n=False,
        return_result=True,
    )
    norm = matplotlib.colors.Normalize(-1.0, 1.0)
    for state, value in (("A", 0.8), ("B", -0.6)):
        ring = result.artists["metric_rings"][state]
        np.testing.assert_allclose(
            ring.get_edgecolor(), plt.get_cmap("coolwarm")(norm(value))
        )
    result.fig.canvas.draw()
    renderer = result.fig.canvas.get_renderer()
    ring = result.artists["metric_rings"]["A"]
    before = ring.get_window_extent(renderer)
    axis = result.axes["map"]
    axis.set(xlim=(-100.0, 100.0), ylim=(-80.0, 80.0))
    result.fig.canvas.draw()
    after = ring.get_window_extent(result.fig.canvas.get_renderer())
    assert before.width > 0
    assert after.width == pytest.approx(before.width)
    assert after.height == pytest.approx(before.height)


@pytest.mark.parametrize("point", [[np.nan, 0.0], [0.0, np.nan], [0.0, np.inf]])
def test_nonfinite_contrast_coordinates_rejected_jointly_before_drawing(
    paga_data, point
):
    paga_data.obsm["X_umap"][1] = point
    _, ax = plt.subplots()
    before = _axes_snapshot(ax)
    with pytest.raises(ValueError, match="(?i)(finite|coordinates|embedding)"):
        _plot(paga_data, ax=ax)
    _assert_axes_untouched(ax, before)


def test_globally_absent_requested_condition_is_an_error(paga_data):
    subset = paga_data[paga_data.obs.condition.ne("treated")].copy()
    with pytest.raises(ValueError, match="(?i)(condition|group|treated)"):
        _plot(subset, graph_node_order=["A", "B", "C", "D"])


def test_incomplete_pie_categories_cannot_hide_omitted_mass(paga_data):
    _, ax = plt.subplots()
    before = _axes_snapshot(ax)
    with pytest.raises(ValueError, match="(?i)(categor|include|observed)"):
        _plot(paga_data, ax=ax, pie_key="condition", pie_categories=["control"])
    _assert_axes_untouched(ax, before)


def test_provided_alignment_known_metadata_is_validated(paga_data):
    table = pd.DataFrame({"state": ["A", "B"], "alignment_cosine": [0.8, -0.6]})
    table.attrs["params"] = {"group1": "unrelated", "rep": "X_pca"}
    with pytest.raises(ValueError, match="does not match"):
        _plot(
            paga_data,
            node_color_mode="alignment",
            alignment_df=table,
            analysis_rep="X_pca",
        )
    table.attrs["params"] = {"group1": "treated", "rep": "X_pca"}
    with pytest.raises(ValueError, match="(?i)(representation|rep)"):
        _plot(
            paga_data,
            node_color_mode="alignment",
            alignment_df=table,
            analysis_rep="X_latent",
        )


def test_velocity_missingness_is_joint_and_usable_count_is_explicit(paga_data):
    # This y value must not survive when the same velocity row has no x value.
    paga_data.obsm["velocity_umap"][0] = [np.nan, 1e9]
    result = _plot(paga_data, show_velocity=True, return_result=True)
    velocity = result.data["velocity"].set_index("state")
    np.testing.assert_allclose(
        velocity.loc["A", ["vx", "vy"]].to_numpy(dtype=float), [0.2, 0.1]
    )
    assert velocity.loc["A", "n_velocity"] == 4


def test_missing_pie_category_stays_distinct_from_literal_na(paga_data):
    paga_data.obs["annotation"] = "known"
    paga_data.obs.loc[paga_data.obs_names[0], "annotation"] = None
    paga_data.obs.loc[paga_data.obs_names[1], "annotation"] = "NA"
    result = _plot(paga_data, pie_key="annotation", return_result=True)
    composition = result.data["composition"].set_index("node")
    assert composition.loc["A", "NA"] == pytest.approx(1 / 5)
    missing_keys = set(composition.columns) - {"known", "NA"}
    assert len(missing_keys) == 1
    assert composition.loc["A", missing_keys.pop()] == pytest.approx(1 / 5)
    assert composition.loc["A", "known"] == pytest.approx(3 / 5)


def test_legend_uses_configured_vector_colors_and_scales(paga_data):
    velocity_color = "#B72275"
    result = _plot(
        paga_data,
        show_velocity=True,
        velocity_color=velocity_color,
        arrow_scale=2.5,
        velocity_scale=11.0,
        legend=True,
        return_result=True,
    )
    legends = [
        child
        for child in result.axes["map"].get_children()
        if isinstance(child, Legend)
    ]
    entries = [
        (label.get_text(), handle)
        for legend in legends
        for label, handle in zip(legend.get_texts(), legend.legend_handles)
    ]
    velocity_entries = [
        (text, handle) for text, handle in entries if "velocity" in text.lower()
    ]
    assert velocity_entries
    assert any("11" in text for text, _ in velocity_entries)
    for _, handle in velocity_entries:
        color = (
            handle.get_color()
            if hasattr(handle, "get_color")
            else handle.get_edgecolor()
        )
        np.testing.assert_allclose(to_rgba(color), to_rgba(velocity_color))
    assert any(
        "2.5" in text and ("shift" in text.lower() or "contrast" in text.lower())
        for text, _ in entries
    )


def test_zero_connectivity_keeps_nodes_and_exports_an_empty_edge_table(
    paga_data, tmp_path
):
    paga_data.uns["paga"]["connectivities"] = sparse.csr_matrix((4, 4), dtype=float)
    result = _plot(paga_data, return_result=True)
    assert result.data["edges"] == []
    assert result.data["edge_table"].empty
    assert {"source", "target", "weight", "displayed", "reason"}.issubset(
        result.data["edge_table"].columns
    )
    assert set(result.artists["reference_nodes"]) == {"A", "B", "C"}
    assert "D" in result.artists["comparison_nodes"]
    assert result.metadata["graph_edges_total"] == 0
    assert result.metadata["graph_edges_displayed"] == 0
    output = result.save(tmp_path / "zero_connectivity", formats="png", dpi=40)
    reloaded = pd.read_csv(output["tables"]["edge_table"])
    assert reloaded.empty
    assert reloaded.columns.tolist() == result.data["edge_table"].columns.tolist()


def test_paga_scgeo_inherits_nondefault_display_velocity_basis(paga_data):
    result = sg.pl.paga_scgeo(
        paga_data,
        node_key="state",
        condition_key="condition",
        group0="control",
        group1="treated",
        basis="pca",
        pie_key=None,
        show_velocity=True,
        min_cells=1,
        return_result=True,
        show=False,
    )
    assert result.metadata["display_basis"] == "X_pca"
    assert result.metadata["velocity_key"] == "velocity_pca"
    velocity = result.data["velocity"].set_index("state")
    np.testing.assert_allclose(
        velocity.loc["A", ["vx", "vy"]].to_numpy(dtype=float), [0.2, 0.1]
    )


def test_long_automatic_labels_wrap_and_manual_offsets_remain_exact():
    from scgeo.pl._paga_layout import place_labels

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set(xlim=(-3, 3), ylim=(-2, 2))
    rows = [("Col13a1-like matrix fibroblast", 0., 0.),
            ("Col14a1-like matrix fibroblast", .12, .04),
            ("Myofibroblast", .08, -.1)]
    artists, table = place_labels(ax, rows, fontsize=9, mode="auto",
                                  offsets={}, obstacles=[(x, y, 11) for _, x, y in rows])
    fig.canvas.draw()
    assert "\n" in artists[rows[0][0]].get_text()
    np.testing.assert_allclose(table[["anchor_x", "anchor_y"]], [(x, y) for _, x, y in rows])
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_bbox_patch().get_window_extent(renderer) for artist in artists.values()]
    assert all(not left.overlaps(right) for i, left in enumerate(boxes) for right in boxes[i + 1:])
    fig2, ax2 = plt.subplots()
    _, fixed = place_labels(ax2, rows[:1], fontsize=9, mode="auto",
                            offsets={rows[0][0]: (20, -40)}, obstacles=[])
    np.testing.assert_allclose(fixed[["offset_x_points", "offset_y_points"]], [[20, -40]])
