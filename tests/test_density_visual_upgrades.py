"""Density semantics: exact counts, comparable scales, and inspectable omissions."""

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
from matplotlib.colors import LogNorm, Normalize

import scgeo as sg


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def count_data():
    pattern = np.array([[.25, .25], [.25, .25], [1.25, .25], [1.25, 1.25]])
    xy = np.vstack([pattern, np.tile(pattern, (10, 1))])
    obs = pd.DataFrame(
        {"group": pd.Categorical(["A"] * 4 + ["B"] * 40,
                                  categories=["B", "A", "empty"])},
        index=[f"cell_{i}" for i in range(len(xy))],
    )
    data = AnnData(np.zeros((len(xy), 1)), obs=obs)
    data.obsm["X_umap"] = xy
    return data


def plot(data, **kwargs):
    options = dict(gridsize=2, grid_extent=(0, 2, 0, 2),
                   groups=["A", "B"], return_result=True, show=False)
    options.update(kwargs)
    return sg.pl.embedding_density(data, "group", **options)


def snapshot(ax):
    return (list(ax.get_children()), ax.get_xlim(), ax.get_ylim(),
            ax.get_title(), ax.figure.get_size_inches().copy())


def assert_unchanged(ax, before):
    assert list(ax.get_children()) == before[0]
    assert ax.get_xlim() == before[1]
    assert ax.get_ylim() == before[2]
    assert ax.get_title() == before[3]
    np.testing.assert_array_equal(ax.figure.get_size_inches(), before[4])


def test_histogram_counts_and_axes_orientation_match_observations(count_data):
    result = plot(count_data, normalize="global")
    grid = result.data["grid"]
    expected = {(0, 0): 2, (1, 0): 1, (0, 1): 0, (1, 1): 1}
    for group, multiplier in [("A", 1), ("B", 10)]:
        subset = grid[grid.group == group]
        assert len(subset) == 4
        for row in subset.itertuples():
            assert row.raw_count == expected[row.ix, row.iy] * multiplier
        image = result.axes[f"density:{group}"].images[0]
        np.testing.assert_allclose(image.get_array().filled(0),
                                   np.array([[2, 1], [0, 1]]) * multiplier)
    assert grid.raw_count.sum() == count_data.n_obs
    assert result.data["cells"].included.all()
    assert all(float(result.axes[f"density:{g}"].get_aspect()) == 1 for g in ["A", "B"])


def test_relative_panels_share_truthful_unit_interval_colorbar(count_data):
    result = plot(count_data, normalize="per_group")
    grid = result.data["grid"]
    first = grid[grid.group == "A"].sort_values(["iy", "ix"])
    second = grid[grid.group == "B"].sort_values(["iy", "ix"])
    np.testing.assert_allclose(first.plotted_value, second.plotted_value)
    assert first.plotted_value.max() == pytest.approx(1)
    for group in ["A", "B"]:
        norm = result.axes[f"density:{group}"].images[0].norm
        assert (norm.vmin, norm.vmax) == (0, 1)
    assert "colorbar" in result.axes
    assert "relative" in result.axes["colorbar"].get_ylabel().lower()


def test_global_panels_use_one_absolute_count_scale(count_data):
    result = plot(count_data, normalize="global")
    for group in ["A", "B"]:
        image = result.axes[f"density:{group}"].images[0]
        assert (image.norm.vmin, image.norm.vmax) == (0, 20)
    a = result.data["grid"].query("group == 'A'").plotted_value.max()
    b = result.data["grid"].query("group == 'B'").plotted_value.max()
    assert b / a == pytest.approx(10)


def test_contour_transforms_once_and_uses_shared_actual_value_levels(count_data):
    result = plot(count_data, gridsize=4, contour=True, normalize="global",
                  smooth_k=1, log1p=True, contour_levels=5)
    grid = result.data["grid"]
    np.testing.assert_allclose(grid.smoothed_count, grid.raw_count)
    np.testing.assert_allclose(grid.transformed_value, np.log1p(grid.raw_count))
    np.testing.assert_allclose(grid.plotted_value, grid.transformed_value)
    levels = result.data["contour_levels"]
    a = levels[levels.group == "A"].sort_values("level_index").value
    b = levels[levels.group == "B"].sort_values("level_index").value
    np.testing.assert_allclose(a, b)
    assert np.diff(a).min() > 0
    for group in ["A", "B"]:
        collections = result.axes[f"density:{group}"].collections
        assert collections
        assert collections[0].norm.vmax == pytest.approx(np.log1p(20))


def test_single_smoothing_step_has_known_impulse_response():
    data = AnnData(np.zeros((1, 1)), obs=pd.DataFrame({"group": ["A"]}, index=["cell"]))
    data.obsm["X_umap"] = np.array([[3.5, 3.5]])
    result = plot(data, groups=["A"], gridsize=7, grid_extent=(0, 7, 0, 7),
                  contour=True, normalize="global", smooth_k=3, log1p=True)
    grid = result.data["grid"]
    expected = ((grid.ix >= 2) & (grid.ix <= 4) &
                (grid.iy >= 2) & (grid.iy <= 4)).astype(float) / 9
    np.testing.assert_allclose(grid.smoothed_count, expected, atol=1e-14)
    np.testing.assert_allclose(grid.transformed_value, np.log1p(expected), atol=1e-14)
    assert grid.raw_count.sum() == 1


def test_log_color_norm_does_not_erase_an_isolated_contour_peak():
    data = AnnData(np.zeros((1, 1)), obs=pd.DataFrame({"group": ["A"]}, index=["cell"]))
    data.obsm["X_umap"] = np.array([[3.5, 3.5]])
    norm = LogNorm(.1, 2)
    result = plot(data, groups=["A"], gridsize=7, grid_extent=(0, 7, 0, 7),
                  contour=True, normalize="global", smooth_k=1, log1p=False, norm=norm)
    grid = result.data["grid"]
    assert grid.displayed.sum() == 1
    assert grid.raw_count.sum() == 1
    collection = result.axes["density:A"].collections[0]
    assert collection.norm is norm
    assert any(len(path.vertices) > 0 for path in collection.get_paths())


def test_view_crop_does_not_change_histograms_or_normalization(count_data):
    full = plot(count_data, normalize="global")
    cropped = plot(count_data, normalize="global", xlim=(.1, .6), ylim=(.1, .6))
    pd.testing.assert_frame_equal(full.data["grid"], cropped.data["grid"])
    for name in ["density:A", "density:B"]:
        assert cropped.axes[name].get_xlim() == pytest.approx((.1, .6))
        assert cropped.axes[name].get_ylim() == pytest.approx((.1, .6))


def test_nonfinite_and_outside_grid_cells_are_retained_and_counted():
    data = AnnData(np.zeros((6, 1)), obs=pd.DataFrame(
        {"group": ["A", "A", "A", "B", "B", None]},
        index=[f"cell_{i}" for i in range(6)]))
    data.obsm["X_umap"] = np.array([[.5, .5], [np.nan, .5], [4, 4],
                                    [1.5, 1.5], [np.inf, 1], [.5, .5]])
    result = plot(data)
    cells = result.data["cells"]
    assert len(cells) == 6
    assert cells.included.tolist() == [True, False, False, True, False, False]
    assert cells.finite_coordinates.tolist() == [True, False, True, True, False, True]
    assert cells.cell.tolist() == data.obs_names.tolist()
    assert result.data["grid"].raw_count.sum() == 2
    summary = result.data["groups"].set_index("group")
    assert summary.loc["A", "n_nonfinite"] == 1
    assert summary.loc["A", "n_outside_grid"] == 1
    assert summary.loc["B", "n_nonfinite"] == 1
    assert len(result.warnings) > 0


def test_declared_category_order_and_empty_panel_are_explicit(count_data):
    result = plot(count_data, groups=None)
    assert result.data["groups"].group.tolist() == ["B", "A", "empty"]
    empty = result.data["groups"].set_index("group").loc["empty"]
    assert empty.raw_count_sum == 0
    assert empty.n_in_grid == 0
    assert not result.data["grid"].query("group == 'empty'").displayed.any()
    assert result.axes["density:empty"].texts


def test_supplied_axes_colorbar_and_norm_keep_caller_geometry(count_data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.subplots_adjust(right=.82)
    cax = fig.add_axes([.87, .2, .025, .6])
    for ax in axes:
        ax.set_xlim(-1, 3)
        ax.set_ylim(-2, 4)
    size = fig.get_size_inches().copy()
    norm = Normalize(0, 50)
    result = plot(count_data, axes=axes, cax=cax, normalize="global", norm=norm)
    assert result.fig is fig
    assert result.axes["colorbar"] is cax
    assert len(fig.axes) == 3
    np.testing.assert_array_equal(fig.get_size_inches(), size)
    for ax in axes:
        assert ax.get_xlim() == (-1, 3)
        assert ax.get_ylim() == (-2, 4)
        assert ax.images[0].norm is norm


def test_owned_density_title_fits_fixed_canvas_without_panel_overlap(count_data):
    result = plot(count_data, figsize=(6.6, 3))
    result.fig.canvas.draw()
    renderer = result.fig.canvas.get_renderer()
    title_box = result.fig._suptitle.get_window_extent(renderer)
    assert title_box.y1 <= result.fig.bbox.y1
    assert title_box.y0 >= result.fig.bbox.y0
    for name in ["density:A", "density:B"]:
        panel_title = result.axes[name].title.get_window_extent(renderer)
        assert panel_title.y1 < title_box.y0


def test_legacy_data_and_subfigure_returns(count_data, tmp_path):
    fig, axes = sg.pl.embedding_density(count_data, "group", groups=["A"], show=False)
    assert len(axes) == 1
    returned = sg.pl.embedding_density(count_data, "group", groups=["A"],
                                      ax=axes[0], return_data=True, show=False)
    assert returned[0] is fig and returned[1][0] is axes[0]
    assert {"cells", "grid", "groups", "contour_levels"} <= set(returned[2])
    parent = plt.figure(figsize=(8, 4))
    subfig = parent.subfigures(1, 2)[0]
    ax = subfig.subplots()
    result = plot(count_data, groups=["A"], ax=ax, show_colorbar=False)
    assert result.fig is parent
    assert result.metadata["export_scope"] == "parent_figure"
    assert "sibling" in result.metadata["source_scope"]
    exported = result.save(tmp_path / "density", formats="svg")
    manifest = json.loads(Path(exported["metadata"]).read_text())
    for key in ["cells", "grid", "groups", "contour_levels"]:
        path = Path(exported["tables"][key])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["tables"][key]["sha256"]
    grid = pd.read_csv(exported["tables"]["grid"])
    assert grid.raw_count.sum() == 4


@pytest.mark.parametrize("options", [
    {"normalize": "unknown"}, {"gridsize": 1}, {"groups": []},
    {"groups": ["absent"]}, {"groups": ["A", "A"]},
    {"grid_extent": (1, 0, 0, 1)}, {"xlim": (1, 0)},
    {"contour": True, "smooth_k": 2},
    {"contour": True, "contour_levels": 1},
    {"contour_level_mode": "unknown"}, {"imshow_alpha": -1},
    {"norm": Normalize(0, 2), "vmin": 0},
])
def test_invalid_inputs_do_not_mutate_supplied_axes(count_data, options):
    _, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    ax.set(title="Caller", xlim=(-3, 3), ylim=(-3, 3))
    before = snapshot(ax)
    settings = {"groups": ["A"], "ax": ax, **options}
    with pytest.raises((ValueError, TypeError)):
        plot(count_data, **settings)
    assert_unchanged(ax, before)


def test_invalid_data_does_not_allocate_an_orphan_figure():
    data = AnnData(np.zeros((1, 1)), obs=pd.DataFrame({"group": ["A"]}, index=["cell"]))
    data.obsm["X_umap"] = np.array([[np.nan, np.nan]])
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="finite"):
        sg.pl.embedding_density(data, "group", show=False)
    assert plt.get_fignums() == before


def test_input_arrays_and_categories_are_not_modified(count_data):
    coordinates = count_data.obsm["X_umap"].copy()
    obs = count_data.obs.copy(deep=True)
    plot(count_data, contour=True, smooth_k=1)
    np.testing.assert_array_equal(count_data.obsm["X_umap"], coordinates)
    pd.testing.assert_frame_equal(count_data.obs, obs)
