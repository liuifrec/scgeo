"""Faceted embeddings: stable scales, subset provenance and finite support."""

from __future__ import annotations

import copy
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import scgeo as sg


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    plt.close("all")


@pytest.fixture
def data():
    rng = np.random.default_rng(213)
    groups = np.repeat(["baseline", "treated", "recovery"], 80)
    labels = np.tile(np.repeat(["A", "B"], 40), 3)
    batches = np.tile(np.repeat(["b1", "b2"], 20), 6)
    points = rng.normal(0, 0.22, (240, 2))
    points += np.where((labels == "A")[:, None], [-1.7, -0.6], [1.9, 0.8])
    vectors = np.zeros_like(points)
    vectors[groups == "baseline"] = [0.3, 0.05]
    vectors[groups == "treated"] = [-0.2, -0.18]
    vectors[groups == "recovery"] = [0.02, 0.32]
    vectors += rng.normal(0, 0.008, vectors.shape)
    obs = pd.DataFrame({
        "condition": pd.Categorical(groups, categories=["treated", "baseline", "recovery"]),
        "batch": batches,
        "cell_type": pd.Categorical(labels, categories=["A", "B"]),
        "score": rng.normal(size=240),
    }, index=[f"cell-{i}" for i in range(240)])
    adata = AnnData(X=np.zeros((240, 1), dtype=float), obs=obs)
    adata.obsm["X_umap"] = points
    adata.obsm["velocity_umap"] = vectors
    adata.uns["cell_type_colors"] = ["#208C8E", "#E99A5F"]
    return adata


def _plot(adata, **kwargs):
    return sg.pl.embedding_facets(adata, return_result=True, show=False, **kwargs)


def test_public_api_and_one_panel_fallback(data):
    result = _plot(data)
    assert list(result.axes) == ["panel_00"]
    assert result.metadata["n_panels"] == 1
    assert result.data["facets"].n_cells.tolist() == [240]
    assert result.data["cells"].displayed.all()
    assert result.data["velocity_grid"].empty
    assert result.metadata["display_basis"] == "X_umap"


def test_categorical_order_and_scanpy_palette_shared(data):
    result = _plot(data, facet="condition", color="cell_type")
    assert result.data["facets"].col_label.tolist() == ["treated", "baseline", "recovery"]
    assert result.metadata["palette_origin"] == "scanpy_uns"
    assert result.metadata["palette"] == {"A": "#208C8E", "B": "#E99A5F"}
    assert all(x.n_cells == 80 for x in result.data["facets"].itertuples())
    assert len(set(tuple(ax.get_xlim()) for ax in result.axes.values())) == 1
    assert len(set(tuple(ax.get_ylim()) for ax in result.axes.values())) == 1


def test_facet_alias_and_explicit_order(data):
    result = _plot(data, facet="condition", facet_order=["baseline", "treated", "recovery"])
    assert result.data["facets"].col_label.tolist() == ["baseline", "treated", "recovery"]
    with pytest.raises(ValueError, match="omits observed"):
        _plot(data, facet="condition", facet_order=["baseline"])
    with pytest.raises(ValueError, match="cannot be combined"):
        _plot(data, facet="condition", facet_col="batch")
    with pytest.raises(ValueError, match="requires facet"):
        _plot(data, facet_order=["baseline"])
    with pytest.raises(ValueError, match="requires facet_row"):
        _plot(data, row_order=["b1"])


def test_two_way_facets_and_derived_facets_are_complete(data):
    result = _plot(data, facet_row="batch", facet_col="condition", color="score")
    assert len(result.axes) == 6
    assert result.data["facets"].n_cells.sum() == 240
    assert result.data["cells"].facet_id.notna().all()
    assert result.metadata["numeric_color_limits"] is not None
    assert len(set(tuple(ax.get_xlim()) for ax in result.axes.values())) == 1
    assert len(set(tuple(ax.get_ylim()) for ax in result.axes.values())) == 1
    assert all(s.n_cells == 40 for s in result.data["facets"].itertuples())


def test_empty_cross_product_panels_are_visible(data):
    data.obs.loc[data.obs.condition == "recovery", "batch"] = "b3"
    result = _plot(data, facet_row="batch", facet_col="condition")
    assert len(result.axes) == 9
    assert (result.data["facets"].n_cells == 0).sum() > 0
    assert result.data["cells"].facet_id.notna().all()
    assert sum(result.data["facets"].n_cells) == 240


def test_truly_missing_facet_distinct_from_literal_missing(data):
    data.obs["condition"] = data.obs.condition.astype(object)
    data.obs.iloc[0, data.obs.columns.get_loc("condition")] = None
    data.obs.iloc[1, data.obs.columns.get_loc("condition")] = "(missing)"
    result = _plot(data, facet="condition")
    assert len(result.axes) == 5
    assert result.metadata["facets"]["missing_labels"]["col"] == "((missing))"
    counts = result.data["facets"].set_index("col_label").n_cells
    assert counts["(missing)"] == 1
    assert counts["((missing))"] == 1


def test_reject_colliding_string_identifiers(data):
    values = data.obs.condition.astype(object)
    values.iloc[0] = 1
    values.iloc[1] = "1"
    data.obs["condition"] = values
    with pytest.raises(ValueError, match="colliding"):
        _plot(data, facet="condition")


def test_shared_velocity_grids_scales_and_source_selection(data):
    result = _plot(data, facet="condition", color="cell_type", velocity="grid",
                   velocity_options={"grid_size": 15, "n_neighbors": 12,
                                     "min_neighbors": 3, "min_coherence": 0.1})
    grids = result.data["velocity_grid"]
    facets = result.data["facets"]
    assert len(grids) == 3 * 15 * 15
    assert (facets.n_grid_supported > 0).all()
    a = grids[grids.facet_id == "panel_00"]
    b = grids[grids.facet_id == "panel_01"]
    np.testing.assert_array_equal(a[["grid_id", "x", "y"]], b[["grid_id", "x", "y"]])
    assert a.loc[a.supported, "vx"].mean() < 0  # treated
    assert b.loc[b.supported, "vx"].mean() > 0  # baseline
    assert result.metadata["velocity_radius_source"] == "pooled_all_facets"
    assert np.isfinite(result.metadata["velocity_scale_resolved"])
    assert result.data["velocity_vectors"].scale.dropna().nunique() == 1
    assert not result.data["velocity_vectors"].empty


def test_shared_numeric_velocity_color_scale(data):
    result = _plot(data, facet="condition", velocity="grid",
                   velocity_options={"grid_size": 13, "min_neighbors": 2,
                                     "n_neighbors": 10, "color_by": "speed"})
    low, high = result.metadata["velocity_shared_color_limits"]
    assert low == 0
    assert high > 0
    assert set(result.data["velocity_grid"].facet_id) == set(result.axes)
    coh = _plot(data, facet="condition", velocity="grid",
                velocity_options={"grid_size": 13, "min_neighbors": 2,
                                  "n_neighbors": 10, "color_by": "coherence"})
    assert coh.metadata["velocity_shared_color_limits"] == [0, 1]


def test_velocity_streamlines_report_real_supported_paths(data):
    result = _plot(data, facet="condition", velocity="stream",
                   velocity_options={"grid_size": 23, "n_neighbors": 12,
                                     "min_neighbors": 3, "stream_density": 0.6})
    grids = result.data["velocity_grid"]
    assert (grids.supported == True).any()
    paths = result.data["velocity_paths"]
    if not paths.empty:
        assert paths.facet_id.isin(result.axes).all()
        assert np.isfinite(paths[["x", "y"]].to_numpy()).all()
    assert result.metadata["interprets_time_or_trajectory"] is False


def test_low_density_and_wrong_velocity_fail_without_drawing(data):
    bad = data.copy()
    bad.obsm["velocity_umap"] = np.zeros((240, 3))
    with pytest.raises(ValueError, match="match"):
        _plot(bad, facet="condition", velocity="grid")
    with pytest.raises(ValueError, match="stream_density"):
        _plot(data, facet="condition", velocity="stream",
              velocity_options={"stream_density": 0.01})
    without = data.copy()
    del without.obsm["velocity_umap"]
    with pytest.raises(ValueError, match="Expected matching projected velocity"):
        _plot(without, velocity="grid")


def test_single_unsupported_facet_no_fabricated_vectors(data):
    # One facet has no jointly finite projected velocities, but all source
    # observations remain in the exported table with an explicit status.
    missing = data.obs.condition == "treated"
    data.obsm["velocity_umap"][missing.to_numpy()] = np.nan
    result = _plot(data, facet="condition", velocity="grid",
                   velocity_options={"grid_size": 13, "min_neighbors": 2, "n_neighbors": 10})
    assert result.data["facets"].iloc[0].n_velocity_eligible == 0
    assert result.data["facets"].iloc[0].n_grid_supported == 0
    assert "no jointly finite projected velocity" in " ".join(result.warnings)
    assert not (result.data["velocity_vectors"].facet_id == "panel_00").any()


def test_downsample_never_changes_velocity_numeric_grid(data):
    opts = {"grid_size": 15, "min_neighbors": 2, "n_neighbors": 12}
    full = _plot(data, facet="condition", velocity="grid", velocity_options=opts)
    small = _plot(data, facet="condition", velocity="grid", velocity_options=opts,
                  max_points=25, random_state=10)
    pd.testing.assert_frame_equal(full.data["velocity_grid"], small.data["velocity_grid"])
    assert small.data["facets"].n_displayed.tolist() == [25, 25, 25]
    assert full.data["facets"].n_displayed.tolist() == [80, 80, 80]


def test_explicit_velocity_multipliers_and_variable_scales(data):
    opts = {"grid_size": 13, "n_neighbors": 10, "min_neighbors": 2, "scale": 0.17}
    result = _plot(data, facet="condition", velocity="grid", velocity_options=opts)
    assert result.metadata["velocity_scale_resolved"] == pytest.approx(0.17)
    assert result.data["velocity_vectors"].scale.dropna().eq(0.17).all()
    varied = _plot(data, facet="condition", velocity="grid", velocity_options={
        "grid_size": 13, "n_neighbors": 10, "min_neighbors": 2,
    }, shared_velocity_scale=False)
    assert varied.metadata["velocity_scale_resolved"] is None
    assert varied.data["velocity_vectors"].scale.dropna().nunique() >= 2


def test_local_limits_differ_if_explicitly_unshared(data):
    mask = data.obs.condition == "treated"
    data.obsm["X_umap"][mask.to_numpy(), 0] += 6
    result = _plot(data, facet="condition", shared_limits=False)
    assert len(set(tuple(ax.get_xlim()) for ax in result.axes.values())) > 1


def test_highlights_are_draw_only_and_exported(data):
    result = _plot(data, facet="condition", color="cell_type", max_points=14,
                   highlight_key="cell_type", highlight="A")
    assert result.data["cells"].highlighted.sum() == 120
    assert result.data["cells"].displayed.sum() == 42
    assert result.data["cells"].loc[result.data["cells"].displayed].groupby("facet_id").size().tolist() == [14] * 3
    with pytest.raises(ValueError, match="Unknown highlight"):
        _plot(data, highlight_key="cell_type", highlight=["absent"])


def test_export_files_reconcile_with_data(data, tmp_path):
    result = _plot(data, facet="condition", color="cell_type", velocity="grid",
                   velocity_options={"grid_size": 12, "n_neighbors": 12,
                                     "min_neighbors": 2})
    result.save(tmp_path / "facets", formats=("png", "svg", "pdf"), dpi=105)
    assert all((tmp_path / f"facets.{extension}").exists() for extension in ("png", "svg", "pdf"))
    manifest = json.loads((tmp_path / "facets.json").read_text())
    assert manifest["plot"]["plot"] == "embedding_facets"
    assert manifest["tables"]
    assert len(result.data["cells"]) == 240
    assert len(result.data["velocity_grid"]) == 3 * 144


def test_no_mutation_of_supplied_ann_data(data):
    before_obs = data.obs.copy(deep=True)
    before_xy = data.obsm["X_umap"].copy()
    before_velocity = data.obsm["velocity_umap"].copy()
    before_uns = copy.deepcopy(data.uns)
    _plot(data, facet_row="batch", facet_col="condition", color="score", velocity="grid",
          velocity_options={"grid_size": 12, "min_neighbors": 2, "n_neighbors": 10})
    pd.testing.assert_frame_equal(data.obs, before_obs)
    np.testing.assert_array_equal(data.obsm["X_umap"], before_xy)
    np.testing.assert_array_equal(data.obsm["velocity_umap"], before_velocity)
    assert data.uns == before_uns


def test_bad_options_and_facet_designs(data):
    with pytest.raises(ValueError, match="cannot be combined"):
        _plot(data, facet="condition", facet_col="batch")
    with pytest.raises(ValueError, match="different observation keys"):
        _plot(data, facet_row="batch", facet_col="batch")
    with pytest.raises(ValueError, match="velocity_options"):
        _plot(data, velocity_options={"n_neighbors": 10})
    with pytest.raises(ValueError, match="Unsupported velocity_options"):
        _plot(data, velocity="grid", velocity_options={"groups": ["baseline"]})
    with pytest.raises(ValueError, match="max_points"):
        _plot(data, max_points=0)
    with pytest.raises(ValueError, match="context"):
        _plot(data, context="all-cells-with-labels")
    with pytest.raises(KeyError, match="facet_col"):
        _plot(data, facet_col="unknown")


def test_multiline_categorical_legend_reserves_a_separate_row(data):
    # Eight real annotation categories, like processed PBMC3k, must never
    # collide with the axes/labels in a wrapped multirow embedding figure.
    labels = [f"state_{i % 8}" for i in range(data.n_obs)]
    data.obs["cell_type"] = pd.Categorical(labels)
    data.uns.pop("cell_type_colors", None)
    result = _plot(data, facet="condition", color="cell_type", col_wrap=2,
                   legend=True)
    fig = result.fig
    assert len(result.axes) == 3
    assert len(fig.axes) == 5  # four grid slots, including one unused, plus legend row
    assert "legend" in result.artists
    legend_ax = fig.axes[-1]
    assert not legend_ax.axison
    fig.canvas.draw()
    assert legend_ax.get_position().y1 < min(
        ax.get_position().y0 for ax in result.axes.values()
    )
    assert len(result.artists["legend"].get_texts()) == 8
