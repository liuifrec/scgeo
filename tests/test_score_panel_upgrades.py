"""Shared score layers must preserve scale, observation identity, and counts."""
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import scgeo as sg


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    obs = pd.DataFrame({"score": pd.array(np.arange(60.), dtype="Float64"),
                        "state": pd.Categorical(["A", "B"] * 30)},
                       index=[f"cell_{i}" for i in range(60)])
    result = AnnData(obs=obs)
    result.obsm["X_umap"] = rng.normal(size=(60, 2))
    result.uns["state_colors"] = ["#3B758C", "#D99252"]
    result.uns["scgeo"] = {"consensus_subspace": {"singular_values": [4., 3., 2., 1.]}}
    return result


@pytest.mark.parametrize("name", ["alignment_panel", "ambiguity_panel", "consensus_subspace_panel"])
def test_one_color_scale_and_full_histogram_under_sampling(data, name):
    data.obs.loc["cell_0", "score"] = pd.NA
    data.obs.loc["cell_1", "score"] = np.inf
    data.obsm["X_umap"][-1] = np.nan  # highest score is still ranked, but not drawn
    before = data.copy()
    function = getattr(sg.pl, name)
    result = function(data, score_key="score", basis="X_umap", topk=6, max_points=12,
                      bins=[0, 20, 40, 60], return_result=True, show=False)
    cells = result.data["cells"]
    assert len(result.fig.axes) == 3  # two panels, exactly one colorbar
    assert result.artists["scores"].norm is result.artists["highlights"].norm
    assert result.artists["scores"].norm.vmin == 2
    assert result.artists["scores"].norm.vmax == 59
    assert cells.selected.sum() == 6 and cells.highlighted.sum() == 5
    assert cells.displayed.sum() == 12
    assert cells.loc[cells.highlighted, "displayed"].all()
    assert cells.histogram_included.sum() == 58
    np.testing.assert_array_equal(result.data["histogram"]["count"], [18, 20, 20])
    pd.testing.assert_frame_equal(before.obs, data.obs)
    np.testing.assert_equal(before.obsm["X_umap"], data.obsm["X_umap"])
    assert before.uns == data.uns
    if name == "consensus_subspace_panel":
        assert result.data["singular_values"].singular_value.tolist() == [4, 3, 2, 1]


def test_score_panel_supplied_axes_and_svg_pdf_png_sources(data, tmp_path):
    fig = plt.figure(figsize=(9, 4))
    left, right = fig.subfigures(1, 2)
    axs = left.subplots(1, 3)
    sibling = right.subplots()
    sibling.plot([0, 1], [1, 0])
    positions = [ax.get_position(original=True).bounds for ax in axs]
    result = sg.pl.alignment_panel(
        data, "score", topk=4, norm=Normalize(-5, 100), show=False,
        axes=dict(zip(["embedding", "histogram", "colorbar"], axs)), return_result=True)
    assert result.fig is fig and len(fig.axes) == 4
    assert result.artists["colorbar"].ax is axs[2]
    np.testing.assert_allclose([ax.get_position(original=True).bounds for ax in axs], positions)
    assert len(sibling.lines) == 1
    output = result.save(tmp_path / "score", formats=("png", "svg", "pdf"), dpi=60)
    manifest = json.loads(Path(output["metadata"]).read_text())
    assert manifest["plot"]["norm"] == {"type": "Normalize", "vmin": -5., "vmax": 100.}
    assert manifest["plot"]["export_scope"] == "parent_figure"
    assert manifest["tables"]["cells"]["rows"] == 60
    assert pd.read_csv(output["tables"]["histogram"])["count"].sum() == 60


@pytest.mark.parametrize("name", ["alignment_panel", "ambiguity_panel", "consensus_subspace_panel"])
def test_legacy_tuple_and_opt_in_tables(data, name):
    function = getattr(sg.pl, name)
    fig, axes = function(data, score_key="score", show=False, topk=0)
    assert isinstance(axes, np.ndarray) and axes.shape == (2,)
    fig2, axes2, tables = function(data, score_key="score", show=False, topk=None, return_data=True)
    assert axes2.shape == axes.shape
    assert not tables["cells"].selected.any()


def test_invalid_panel_inputs_fail_before_drawing(data):
    fig, axs = plt.subplots(1, 2)
    axes = {"embedding": axs[0], "histogram": axs[1]}
    with pytest.raises(ValueError, match="colorbar"):
        sg.pl.alignment_panel(data, "score", axes=axes, show=False)
    with pytest.raises(ValueError, match="max_points"):
        sg.pl.alignment_panel(data, "score", axes=axes, colorbar=False,
                              topk=10, max_points=5, show=False)
    with pytest.raises(ValueError, match="topk"):
        sg.pl.ambiguity_panel(data, "score", topk=-1, show=False)
    with pytest.raises(TypeError, match="numeric"):
        sg.pl.alignment_panel(data, "state", show=False)
    assert len(plt.get_fignums()) == 1
    assert not axs[0].collections and not axs[1].patches


def test_all_missing_scores_have_no_invented_highlights(data):
    data.obs["score"] = np.nan
    result = sg.pl.ambiguity_panel(data, "score", return_result=True, show=False)
    assert result.data["histogram"]["count"].sum() == 0
    assert not result.data["cells"].highlighted.any()
    assert result.metadata["histogram_n"] == 0
    assert len(result.fig.axes) == 3


def test_standalone_highlights_retain_selection_and_accept_shared_scale(data):
    data.obsm["X_umap"][-1] = np.nan
    result = sg.pl.highlight_topk_cells(
        data, "score", topk=3, background=False, add_colorbar=False,
        norm=Normalize(0, 80), return_result=True, show=False)
    assert result.data.selected.sum() == 3
    assert result.data.highlighted.sum() == result.data.displayed.sum() == 2
    assert result.metadata["basis"] == "X_umap"
    assert result.metadata["norm"]["vmax"] == 80
    assert result.warnings
    assert len(result.fig.axes) == 1
    fig, ax, cells = sg.pl.highlight_topk_cells(data, "score", topk=0, return_data=True, show=False)
    assert not cells.selected.any()


def test_score_embedding_uses_scanpy_palette_and_explicit_colorbar(data):
    before = copy.deepcopy(data.uns)
    categorical = sg.pl.score_umap(data, "state", legend=False, return_result=True, show=False)
    assert categorical.metadata["palette_origin"] == "scanpy_uns"
    assert categorical.metadata["palette"] == dict(zip(["A", "B"], data.uns["state_colors"]))
    assert categorical.axes["embedding"].get_legend() is None
    fig, axs = plt.subplots(1, 2)
    numeric = sg.pl.score_embedding(data, "score", ax=axs[0], cax=axs[1], return_result=True, show=False)
    assert numeric.artists["colorbar"].ax is axs[1]
    assert len(fig.axes) == 2
    assert data.uns == before
