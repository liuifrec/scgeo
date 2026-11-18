"""Observation identity, missing evidence, palettes, and spatial contour checks."""
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import scgeo as sg
from scgeo.pl._ood_landscape import _compute_density_grid


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def data():
    adata = AnnData(obs=pd.DataFrame({
        "map_pred": pd.Categorical(["A", "novel", "nan", None, "A", "novel", "A", "A"]),
        "reference": pd.Categorical(["A"] * 8),
        "condition": ["query"] * 6 + ["reference"] * 2,
        "map_confidence": pd.array([.5, .1, None, .1, .3, .9, .8, .7], dtype="Float64"),
        "map_ood_score": pd.array([.2, .9, .7, None, .1, .8, .3, .2], dtype="Float64"),
    }, index=[f"cell_{i}" for i in range(8)]))
    adata.obsm["X_umap"] = np.column_stack((np.arange(8.), np.arange(8.) ** 2))
    adata.obsm["X_umap"][1] = np.nan
    adata.uns["reference_colors"] = ["#123456"]
    return adata


def test_confidence_stable_low_selection_full_identity_and_axes(data):
    fig, axs = plt.subplots(1, 3)
    bounds = [axis.get_position().bounds for axis in axs]
    result = sg.pl.mapping_confidence_umap(data, basis="X_umap", highlight_low_k=2,
        ax=axs[0], cax=axs[1], return_result=True, show=False)
    assert result.fig is fig
    assert result.data.loc[result.data.selected_low, "obs_position"].tolist() == [1, 3]
    assert result.data.loc[result.data.highlighted, "obs_position"].tolist() == [3]
    assert result.artists["scores"].norm.vmin == .1
    assert result.artists["colorbar"].ax is axs[1]
    np.testing.assert_allclose([axis.get_position().bounds for axis in axs], bounds)
    legacy = sg.pl.mapping_confidence_umap(data, highlight_low_k=0, colorbar=False, show=False)
    assert len(legacy) == 2 and legacy[1].shape == (1,)


def test_ood_no_threshold_is_unknown_and_explicit_threshold_is_auditable(data, tmp_path):
    result = sg.pl.ood_cells(data, basis="X_umap", show_only_flagged=True,
                              colorbar=False, return_result=True, show=False)
    assert result.data.flagged.isna().all()
    assert result.metadata["threshold_source"] == "not_supplied"
    assert result.data.displayed.sum() == 7
    flagged = sg.pl.ood_cells(data, threshold=.7, show_only_flagged=True,
                               return_result=True, show=False)
    assert flagged.data.loc[flagged.data.displayed, "obs_position"].tolist() == [2, 5]
    assert pd.isna(flagged.data.flagged.iloc[3])
    assert "colorbar" not in flagged.artists
    output = flagged.save(tmp_path / "ood", formats="svg")
    exported = pd.read_csv(output["tables"]["data"])
    assert len(exported) == data.n_obs and exported.flagged.isna().sum() == 1


def test_mapping_novel_predictions_and_missing_labels_are_retained(data):
    before = data.obs.copy(deep=True)
    uns = copy.deepcopy(data.uns)
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    bounds = [axis.get_position().bounds for axis in axs]
    names = ["prediction", "confidence", "ood", "confidence_colorbar", "ood_colorbar"]
    result = sg.pl.mapping_qc_panel(data, basis="X_umap", palette_from="reference",
        condition_key="condition", query_value="query", axes=dict(zip(names, axs)),
        return_result=True, show=False)
    assert result.fig is fig and len(fig.axes) == 6
    legend = result.data["legend"].set_index("group")
    assert set(legend.index) == {"A", "novel", "nan", "(missing)"}
    assert legend["count"].sum() == 6
    assert legend.loc["novel", "count"] == 2
    assert result.metadata["palette"]["A"] == "#123456"
    assert len(result.artists["prediction:novel"].get_offsets()) == 1
    cells = result.data["cells"]
    assert cells.prediction_displayed.sum() == 5
    assert cells.reference_displayed.sum() == 2
    assert cells.scalar_displayed.sum() == 7
    assert pd.isna(cells.prediction.iloc[3]) and cells.prediction_label.iloc[3] == "(missing)"
    np.testing.assert_allclose([axis.get_position().bounds for axis in axs], bounds)
    pd.testing.assert_frame_equal(data.obs, before)
    assert data.uns == uns
    fig2, axes2, legend2 = sg.pl.mapping_qc_panel(data, colorbar=False, return_legend_data=True, show=False)
    assert axes2.shape == (3,) and sum(row["count"] for row in legend2) == 8


def test_mapping_cax_errors_do_not_draw_and_colliding_categories_are_rejected(data):
    fig, axs = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="missing panels"):
        sg.pl.mapping_qc_panel(data, axes=dict(zip(["prediction", "confidence", "ood"], axs)), show=False)
    assert all(not axis.collections for axis in axs)
    data.obs["map_pred"] = [1, "1", "A", "A", "A", "A", "A", "A"]
    with pytest.raises(ValueError, match="colliding"):
        sg.pl.mapping_qc_panel(data, show=False)


def test_contour_bins_keep_xy_orientation_and_export_support():
    # Four bins have distinct means; x and y ranges deliberately differ.
    xy = np.repeat([[0., 0.], [4., 0.], [0., 2.], [4., 2.]], 3, axis=0)
    values = np.repeat([1., 2., 3., 4.], 3)
    X, Y, Z = _compute_density_grid(xy[:, 0], xy[:, 1], values, gridsize=2)
    np.testing.assert_equal(X, [[1., 3.], [1., 3.]])
    np.testing.assert_equal(Y, [[.5, .5], [1.5, 1.5]])
    np.testing.assert_equal(Z, [[1., 2.], [3., 4.]])
    adata = AnnData(obs=pd.DataFrame({"ood": values}, index=[f"c{i}" for i in range(12)]))
    adata.obsm["X_display"] = xy
    result = sg.pl.ood_landscape(adata, ood_key="ood", basis="display", contour_gridsize=2,
        contour_quantile=.5, return_result=True, show=False)
    grid = result.data["contour_grid"]
    np.testing.assert_equal(grid.mean_score.to_numpy().reshape(2, 2), Z)
    assert grid["count"].sum() == 12 and grid.supported.all()
    assert result.metadata["contour_status"] == "drawn"
    assert result.metadata["contour_levels"] == [2.5]


def test_landscape_summary_axes_missing_scores_and_contour_omissions(data, tmp_path):
    fig, axs = plt.subplots(1, 4)
    bounds = [axis.get_position().bounds for axis in axs]
    result = sg.pl.ood_landscape(data, ood_key="map_ood_score", basis="X_umap",
        groupby="map_pred", threshold=.7, contour_gridsize=10,
        axes={"landscape": axs[0], "summary": axs[1], "colorbar": axs[2]},
        top_n_groups=2, return_result=True, show=False)
    assert result.fig is fig and len(fig.axes) == 4
    assert result.data["cells"].flagged.sum() == 3
    assert result.data["cells"].outline_displayed.sum() == 2
    assert result.data["summary"].n.sum() == data.n_obs
    assert result.data["summary"].displayed.sum() == 2
    assert result.metadata["contour_status"] == "no_supported_bins"
    assert result.warnings and not result.data["contour_grid"].supported.any()
    assert result.data["contour_grid"]["count"].sum() == 6
    np.testing.assert_allclose([axis.get_position().bounds for axis in axs], bounds)
    output = result.save(tmp_path / "landscape", formats="svg")
    meta = json.loads(Path(output["metadata"]).read_text())
    assert meta["plot"]["export_scope"] == "parent_figure"
    assert len(meta["tables"]) == 4
