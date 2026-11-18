"""Legacy summary renderers: exact sources, owned layouts, and density fallback."""
import copy
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
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
    obs = pd.DataFrame({"condition": pd.Categorical(["A", "B"] * 10),
                        "score": np.arange(20.)}, index=[f"c{i}" for i in range(20)])
    adata = AnnData(obs=obs)
    adata.obsm["X_umap"] = np.random.default_rng(4).normal(size=(20, 2))
    adata.uns["condition_colors"] = ["#278C84", "#B56C50"]
    adata.uns["scgeo"] = {
        "shift": {"by": {"A": {"delta_norm": 2.}, "B": {"delta_norm": .5}, "C": {"delta_norm": np.nan}}},
        "density_overlap": {"global": {"bc": .5, "hellinger": .7},
                            "by": {"A": {"bc": .8, "hellinger": .4}, "B": {"bc": .2, "hellinger": np.nan}}},
        "distribution_test": {"global": {"stat": 8., "p_perm": .02, "n_perm": 99},
                              "by": {"A": {"stat": 4., "p_perm": .1}, "B": {"stat": 10., "p_perm": .02}}}}
    return adata


@pytest.mark.parametrize("name,names", [("density_overlap", ["bc", "hellinger"]),
                                        ("distribution_test", ["summary"])])
def test_summary_sources_and_caller_axes(data, name, names):
    fig, axs = plt.subplots(1, len(names) + 1, figsize=(10, 4))
    before = [ax.get_position().bounds for ax in axs]
    fig.suptitle("Caller title")
    result = getattr(sg.pl, name)(data, axes=dict(zip(names, axs)), return_result=True, show=False)
    assert result.fig is fig and set(result.axes) == set(names)
    assert fig._suptitle.get_text() == "Caller title"
    np.testing.assert_allclose([ax.get_position().bounds for ax in axs], before)
    assert len(result.data["source"]) == 2
    assert list(result.data["displayed"].index) == ["B", "A"]
    assert result.data["source"].equals(pd.DataFrame.from_dict(data.uns["scgeo"][name]["by"], orient="index").rename_axis("group"))


def test_global_separate_metric_axes_and_default_shape(data):
    result = sg.pl.distribution_test(data, level="global", separate_metrics=True,
                                     return_result=True, show=False)
    assert set(result.axes) == {"statistic", "probability"}
    assert result.data["displayed"].value.tolist() == [8., .02]
    assert result.axes["statistic"].patches[0].get_height() == 8.
    assert result.axes["probability"].patches[0].get_height() == .02
    assert not result.warnings
    fig, axes = sg.pl.distribution_test(data, level="global", show=False)
    assert len(axes) == 1
    density = sg.pl.density_overlap(data, level="global", return_result=True, show=False)
    assert density.data["displayed"].value.tolist() == [.5, .7]


def test_delta_ranking_exports_missing_and_does_not_layout_parent(data):
    fig, axs = plt.subplots(1, 2)
    before = axs[1].get_position().bounds
    result = sg.pl.delta_rank(data, ax=axs[0], return_result=True)
    assert result.axes["rank"] is axs[0]
    assert result.data["ranked"].delta_norm.iloc[:2].tolist() == [2., .5]
    assert result.data["ranked"].displayed.tolist() == [True, True, False]
    assert axs[1].get_position().bounds == before
    assert isinstance(sg.pl.delta_rank(data, top_n=1), matplotlib.axes.Axes)


def test_density_fallback_with_no_uns_preserves_axis_and_basis(data):
    data.uns.clear()
    data.obs["density_overlap"] = data.obs.score
    fig, axs = plt.subplots(1, 2)
    result = sg.pl.density_overlap_grid(data, basis="X_umap", ax=axs[0], colorbar=False,
                                       return_result=True, show=False)
    assert result.fig is fig and len(fig.axes) == 2
    assert result.axes["embedding"] is axs[0]
    assert result.metadata["source_mode"] == "observation_score"
    assert len(result.data) == data.n_obs


def test_density_grid_exports_pixel_centers_values_and_missingness(data, tmp_path):
    field = np.array([[1., np.nan, 2.], [3., 4., 5.]])
    data.uns["scgeo"]["density_overlap"] = {"overlap": field,
        "grid": {"xmin": 0., "xmax": 6., "ymin": -2., "ymax": 2.}}
    fig, axs = plt.subplots(1, 2)
    result = sg.pl.density_overlap_grid(data, ax=axs[0], cax=axs[1], return_result=True, show=False)
    grid = result.data["grid"]
    np.testing.assert_equal(grid.value.to_numpy().reshape(2, 3), field)
    np.testing.assert_equal(grid.x.unique(), [1., 3., 5.])
    np.testing.assert_equal(grid.y.unique(), [-1., 1.])
    assert result.artists["colorbar"].ax is axs[1]
    output = result.save(tmp_path / "density", formats=("svg", "png"), dpi=60)
    assert pd.read_csv(output["tables"]["grid"]).value.isna().sum() == 1
    assert json.loads(Path(output["metadata"]).read_text())["plot"]["extent"] == [0, 6, -2, 2]


def test_report_source_counts_and_legend_match_actual_categories(data):
    before = copy.deepcopy(data.uns)
    fig, axs = plt.subplots(2, 2)
    result = sg.pl.distribution_report(data, score_key="score", axes=dict(zip(
        ["embedding", "density", "test", "score"], axs.flat)), return_result=True, show=False)
    assert result.fig is fig and len(fig.axes) == 4
    assert result.data["histogram"]["count"].sum() == data.n_obs
    assert len(result.data["summary_text"]) == 2
    assert result.metadata["palette"] == {"A": "#278C84", "B": "#B56C50"}
    for label in ("A", "B"):
        np.testing.assert_allclose(result.artists["embedding"][label].get_facecolors()[0][:3],
                                   to_rgba(result.metadata["palette"][label])[:3])
    assert data.uns == before
