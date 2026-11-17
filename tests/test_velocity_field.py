"""Scientific support, provenance, and composition contracts for velocity fields."""

from __future__ import annotations

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
from matplotlib.colors import Normalize
from scipy import sparse

import scgeo as sg


def _make_data(points, velocities, conditions=None):
    points = np.asarray(points, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if conditions is None:
        conditions = [
            "control" if i % 2 == 0 else "treated" for i in range(len(points))
        ]
    obs = pd.DataFrame(
        {"condition": pd.Categorical(conditions)},
        index=[f"cell_{i:03d}" for i in range(len(points))],
    )
    data = AnnData(np.zeros((len(points), 1)), obs=obs)
    data.obsm["X_umap"] = points
    data.obsm["velocity_umap"] = velocities
    data.obsm["X_pca"] = np.column_stack([points, np.arange(len(points))])
    data.obsm["velocity_pca"] = np.column_stack([velocities, np.zeros(len(points))])
    return data


@pytest.fixture
def uniform_data():
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 7), np.linspace(-1.0, 1.0, 7))
    points = np.column_stack([x.ravel(), y.ravel()])
    yield _make_data(points, np.tile([1.0, 0.25], (len(points), 1)))
    plt.close("all")


def _plot(data, **kwargs):
    options = {
        "basis": "umap",
        "grid_size": 11,
        "n_neighbors": 16,
        "radius": 0.75,
        "min_neighbors": 3,
        "show": False,
    }
    options.update(kwargs)
    return sg.pl.velocity_field(data, **options)


def _snapshot(ax):
    return {
        "children": list(ax.get_children()),
        "xlim": ax.get_xlim(),
        "ylim": ax.get_ylim(),
        "title": ax.get_title(),
    }


def _assert_untouched(ax, before):
    assert list(ax.get_children()) == before["children"]
    assert ax.get_xlim() == before["xlim"]
    assert ax.get_ylim() == before["ylim"]
    assert ax.get_title() == before["title"]


def test_uniform_field_recovers_vector_speed_and_directional_agreement(uniform_data):
    result = _plot(uniform_data, return_result=True)
    grid = result.data["grid"]
    supported = grid.loc[grid.supported]
    assert len(supported) > 10
    np.testing.assert_allclose(
        supported[["vx", "vy"]], np.tile([1.0, 0.25], (len(supported), 1))
    )
    np.testing.assert_allclose(supported.speed, np.hypot(1.0, 0.25))
    np.testing.assert_allclose(supported.mean_speed, supported.speed)
    np.testing.assert_allclose(supported.coherence, 1.0)
    assert grid.radius.eq(0.75).all()
    assert supported.n_neighbors.ge(3).all()
    assert supported.n_neighbors.le(16).all()
    assert result.metadata["coherence_is_confidence"] is False


def test_opposed_colocated_velocities_reduce_coherence_without_invented_direction():
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 5))
    points = np.repeat(np.column_stack([x.ravel(), y.ravel()]), 2, axis=0)
    opposite = _make_data(points, np.tile([[1.0, 0.0], [-1.0, 0.0]], (25, 1)))
    result = _plot(
        opposite,
        n_neighbors=len(points),
        radius=0.7,
        min_coherence=0.2,
        return_result=True,
    )
    grid = result.data["grid"]
    has_neighbors = grid.n_neighbors.ge(3)
    assert has_neighbors.any()
    np.testing.assert_allclose(grid.loc[has_neighbors, ["vx", "vy"]], 0.0, atol=1e-14)
    np.testing.assert_allclose(grid.loc[has_neighbors, "coherence"], 0.0, atol=1e-14)
    np.testing.assert_allclose(grid.loc[has_neighbors, "mean_speed"], 1.0)
    assert not grid.supported.any()
    assert not result.data["vectors"].displayed.any()
    assert not result.artists["grid_arrows"]
    plt.close(result.fig)


def test_coherence_uses_raw_vector_mean_relative_to_mean_speed():
    # Equal weights for +3 and -1 give mean velocity +1, mean speed 2, and
    # coherence 0.5. Averaging unit directions would incorrectly give zero.
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 5))
    points = np.repeat(np.column_stack([x.ravel(), y.ravel()]), 2, axis=0)
    data = _make_data(points, np.tile([[3.0, 0.0], [-1.0, 0.0]], (25, 1)))
    result = _plot(data, n_neighbors=len(points), radius=0.7, return_result=True)
    supported = result.data["grid"].loc[lambda frame: frame.supported]
    assert not supported.empty
    np.testing.assert_allclose(
        supported[["vx", "vy"]], np.tile([1.0, 0.0], (len(supported), 1))
    )
    np.testing.assert_allclose(supported.mean_speed, 2.0)
    np.testing.assert_allclose(supported.coherence, 0.5)
    plt.close(result.fig)


@pytest.mark.parametrize("mode", ["grid", "stream"])
def test_finite_radius_leaves_disconnected_cloud_gap_unsupported(mode):
    local_x, local_y = np.meshgrid(np.linspace(-0.4, 0.4, 7), np.linspace(-0.6, 0.6, 9))
    local = np.column_stack([local_x.ravel(), local_y.ravel()])
    points = np.vstack([local + [-3.0, 0.0], local + [3.0, 0.0]])
    data = _make_data(points, np.tile([1.0, 0.0], (len(points), 1)))
    result = _plot(
        data,
        mode=mode,
        grid_size=(65, 25),
        n_neighbors=30,
        radius=0.55,
        min_neighbors=3,
        return_result=True,
    )
    grid = result.data["grid"]
    gap = grid.x.abs().lt(1.5)
    assert gap.any()
    assert not grid.loc[gap, "supported"].any()
    assert grid.loc[gap, "n_neighbors"].eq(0).all()
    assert grid.loc[~gap, "supported"].any()
    if mode == "grid":
        shown = result.data["vectors"].loc[lambda frame: frame.displayed]
        assert not shown.source_x.abs().lt(1.5).any()
    else:
        paths = result.data["paths"]
        assert not paths.empty
        assert not paths.x.abs().lt(1.5).any()
        for _, path in paths.groupby(["path_id", "segment_id"]):
            assert not (path.x.min() < -1.5 and path.x.max() > 1.5)
        segments = result.artists["stream_lines"].get_segments()
        assert len(segments)
        for segment in segments:
            assert np.isfinite(segment).all()
            assert not (np.abs(segment[:, 0]) < 1.5).any()
            assert not (segment[:, 0].min() < -1.5 and segment[:, 0].max() > 1.5)
    plt.close(result.fig)


def test_condition_selection_excludes_unrelated_large_vectors(uniform_data):
    points = np.vstack([uniform_data.obsm["X_umap"], uniform_data.obsm["X_umap"]])
    velocities = np.vstack(
        [
            uniform_data.obsm["velocity_umap"],
            np.tile([1000.0, -1000.0], (uniform_data.n_obs, 1)),
        ]
    )
    conditions = (
        uniform_data.obs.condition.astype(str).tolist()
        + ["unrelated"] * uniform_data.n_obs
    )
    data = _make_data(points, velocities, conditions)
    result = _plot(
        data,
        condition_key="condition",
        groups=["control", "treated"],
        return_result=True,
    )
    supported = result.data["grid"].loc[lambda frame: frame.supported]
    np.testing.assert_allclose(
        supported[["vx", "vy"]], np.tile([1.0, 0.25], (len(supported), 1))
    )
    cells = result.data["cells"]
    assert len(cells) == data.n_obs
    assert not cells.loc[cells.condition.eq("unrelated"), "selected"].any()
    assert not cells.loc[cells.condition.eq("unrelated"), "eligible"].any()
    assert cells.loc[cells.condition.isin(["control", "treated"]), "eligible"].all()


def test_joint_invalid_row_exclusion_retains_complete_source_provenance(uniform_data):
    uniform_data.obsm["X_umap"][0] = [np.nan, 0.0]
    uniform_data.obsm["velocity_umap"][1] = [np.nan, 1e9]
    uniform_data.obsm["velocity_umap"][2] = [1e9, np.inf]
    original_obs = uniform_data.obs.copy(deep=True)
    original_xy = uniform_data.obsm["X_umap"].copy()
    original_velocity = uniform_data.obsm["velocity_umap"].copy()
    result = _plot(uniform_data, return_result=True)
    cells = result.data["cells"].set_index("obs_position")
    assert len(cells) == uniform_data.n_obs
    assert cells.cell.tolist() == uniform_data.obs_names.tolist()
    assert not cells.loc[[0, 1, 2], "eligible"].any()
    assert not cells.loc[0, "finite_coordinates"]
    assert not cells.loc[[1, 2], "finite_velocity"].any()
    assert np.isnan(cells.loc[0, "x"])
    assert np.isnan(cells.loc[1, "vx"])
    assert np.isinf(cells.loc[2, "vy"])
    valid = result.data["grid"].loc[lambda frame: frame.supported]
    np.testing.assert_allclose(
        valid[["vx", "vy"]], np.tile([1.0, 0.25], (len(valid), 1))
    )
    pd.testing.assert_frame_equal(uniform_data.obs, original_obs)
    np.testing.assert_array_equal(uniform_data.obsm["X_umap"], original_xy)
    np.testing.assert_array_equal(uniform_data.obsm["velocity_umap"], original_velocity)


@pytest.mark.parametrize("problem", ["wrong_basis", "wrong_width", "missing_velocity"])
def test_basis_and_shape_validation_precedes_axes_mutation(uniform_data, problem):
    options = {}
    if problem == "wrong_basis":
        options["velocity_basis"] = "pca"
    elif problem == "wrong_width":
        uniform_data.obsm["velocity_umap"] = np.zeros((uniform_data.n_obs, 3))
    else:
        del uniform_data.obsm["velocity_umap"]
    _, ax = plt.subplots()
    ax.set(title="Caller panel", xlim=(-4, 4), ylim=(-3, 3))
    before = _snapshot(ax)
    with pytest.raises(
        (ValueError, KeyError), match="(?i)(velocity|basis|shape|embedding)"
    ):
        _plot(uniform_data, ax=ax, **options)
    _assert_untouched(ax, before)


def test_display_sampling_and_cropping_change_no_field_numbers(uniform_data):
    options = {"background_key": "condition", "radius": None, "return_result": True}
    complete = _plot(uniform_data, **options)
    sampled = _plot(
        uniform_data,
        max_points=12,
        random_state=7,
        required_indices=[0, 1],
        xlim=(-0.5, 0.5),
        ylim=(-0.5, 0.5),
        **options,
    )
    repeat = _plot(
        uniform_data,
        max_points=12,
        random_state=7,
        required_indices=[0, 1],
        xlim=(-0.5, 0.5),
        ylim=(-0.5, 0.5),
        **options,
    )
    pd.testing.assert_frame_equal(complete.data["grid"], sampled.data["grid"])
    pd.testing.assert_frame_equal(complete.data["vectors"], sampled.data["vectors"])
    pd.testing.assert_frame_equal(sampled.data["cells"], repeat.data["cells"])
    cells = sampled.data["cells"]
    assert len(cells) == uniform_data.n_obs
    assert cells.displayed.sum() == 12
    assert cells.loc[cells.obs_position.isin([0, 1]), "displayed"].all()
    assert sampled.axes["field"].get_xlim() == (-0.5, 0.5)
    assert sampled.axes["field"].get_ylim() == (-0.5, 0.5)


@pytest.mark.parametrize("mode", ["grid", "stream"])
@pytest.mark.parametrize("case", ["zero_velocity", "insufficient_neighbors"])
def test_zero_or_insufficient_field_draws_no_invented_directions(
    uniform_data, mode, case
):
    options = {"mode": mode, "return_result": True}
    if case == "zero_velocity":
        uniform_data.obsm["velocity_umap"][:] = 0.0
    else:
        options.update(radius=0.001, min_neighbors=3)
    result = _plot(uniform_data, **options)
    assert not result.data["grid"].supported.any()
    if mode == "grid":
        assert not result.data["vectors"].displayed.any()
        assert not result.artists["grid_arrows"]
    else:
        assert result.data["paths"].empty
        assert not result.artists["stream_arrows"]
    assert result.data["grid"].reason.notna().all()


def test_scaled_vector_table_matches_raw_field_geometry(uniform_data):
    result = _plot(uniform_data, scale=2.75, return_result=True)
    vectors = result.data["vectors"].loc[lambda frame: frame.displayed]
    assert not vectors.empty
    assert vectors.scale.eq(2.75).all()
    assert result.metadata["scale_resolved"] == 2.75
    np.testing.assert_allclose(vectors.target_x - vectors.source_x, 2.75 * vectors.vx)
    np.testing.assert_allclose(vectors.target_y - vectors.source_y, 2.75 * vectors.vy)
    grid = result.data["grid"].set_index("grid_id")
    np.testing.assert_allclose(
        vectors[["vx", "vy"]], grid.loc[vectors.grid_id, ["vx", "vy"]]
    )


def test_return_contracts_and_caller_limits(uniform_data):
    fig, ax = _plot(uniform_data)
    assert ax.figure is fig
    assert float(ax.get_aspect()) == pytest.approx(1.0)
    supplied_fig, supplied_ax = plt.subplots()
    supplied_ax.set(xlim=(-2.0, 3.0), ylim=(-3.0, 2.0))
    output = _plot(uniform_data, ax=supplied_ax)
    assert output is supplied_ax
    assert supplied_ax.figure is supplied_fig
    assert supplied_ax.get_xlim() == (-2.0, 3.0)
    assert supplied_ax.get_ylim() == (-3.0, 2.0)
    fig_data, ax_data, data = _plot(uniform_data, return_data=True)
    assert ax_data.figure is fig_data
    assert {"cells", "grid", "vectors"}.issubset(data)
    result = _plot(uniform_data, return_result=True)
    assert isinstance(result, sg.pl.PlotResult)
    assert result.axes["field"].figure is result.fig


def test_numeric_color_uses_shared_norm_and_supplied_colorbar_axes(uniform_data):
    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [8, 1]})
    shared_norm = Normalize(0.0, 2.0)
    before_size = fig.get_size_inches().copy()
    result = _plot(
        uniform_data,
        ax=ax,
        cax=cax,
        color_by="speed",
        cmap="viridis",
        norm=shared_norm,
        xlim=(-2.0, 2.0),
        ylim=(-1.5, 1.5),
        return_result=True,
    )
    assert result.fig is fig
    assert result.axes["field"] is ax
    assert result.axes["colorbar"] is cax
    assert len(fig.axes) == 2
    np.testing.assert_array_equal(fig.get_size_inches(), before_size)
    assert ax.get_xlim() == (-2.0, 2.0)
    assert ax.get_ylim() == (-1.5, 1.5)
    assert result.artists["colorbar"].mappable.norm is shared_norm
    expected_color = plt.get_cmap("viridis")(shared_norm(np.hypot(1.0, 0.25)))
    np.testing.assert_allclose(
        result.artists["grid_arrows"][0].get_edgecolor()[:3], expected_color[:3]
    )


def test_numeric_streamline_colors_follow_the_scalar_after_canvas_draw(uniform_data):
    result = _plot(
        uniform_data,
        mode="stream",
        color_by="coherence",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.6,
        return_result=True,
    )
    result.fig.canvas.draw()
    collection = result.artists["stream_lines"]
    colors = collection.get_colors()
    assert len(colors) > 0
    expected = plt.get_cmap("viridis")(1.0)
    np.testing.assert_allclose(colors[:, :3], np.tile(expected[:3], (len(colors), 1)))
    np.testing.assert_allclose(colors[:, 3], 0.6)
    np.testing.assert_allclose(collection.get_array(), 1.0)
    for arrow in result.artists["stream_arrows"]:
        np.testing.assert_allclose(arrow.get_edgecolor()[:3], expected[:3])


@pytest.mark.parametrize("mode", ["grid", "stream"])
def test_plot_source_export_roundtrip_and_hashes(uniform_data, tmp_path, mode):
    result = _plot(
        uniform_data,
        mode=mode,
        condition_key="condition",
        groups=["control", "treated"],
        background_key="condition",
        color_by="speed",
        max_points=15,
        random_state=3,
        return_result=True,
    )
    output = result.save(tmp_path / mode, formats=("png", "svg"), dpi=50)
    metadata = json.loads(Path(output["metadata"]).read_text())
    keys = ["cells", "grid", "vectors" if mode == "grid" else "paths"]
    for key in keys:
        source = Path(output["tables"][key])
        loaded = pd.read_csv(source)
        pd.testing.assert_frame_equal(loaded, result.data[key], check_dtype=False)
        assert (
            metadata["tables"][key]["sha256"]
            == hashlib.sha256(source.read_bytes()).hexdigest()
        )
    for extension, path in output["figures"].items():
        assert (
            metadata["figures"][extension]["sha256"]
            == hashlib.sha256(Path(path).read_bytes()).hexdigest()
        )
    assert len(result.data["cells"]) == uniform_data.n_obs
    assert metadata["plot"]
    json.dumps(result.metadata, allow_nan=False)


def test_stream_source_matches_actual_rendered_segments(uniform_data):
    result = _plot(uniform_data, mode="stream", return_result=True)
    paths = result.data["paths"]
    segments = result.artists["stream_lines"].get_segments()
    assert len(segments) > 0
    saved_segments = [
        group.sort_values("vertex")[["x", "y"]].to_numpy()
        for _, group in paths.groupby(["path_id", "segment_id"], sort=False)
    ]
    assert len(saved_segments) == len(segments)
    for source, actual in zip(saved_segments, segments):
        np.testing.assert_allclose(source, actual)


def test_subfigure_exports_parent_canvas_without_changing_sibling(
    uniform_data, tmp_path
):
    fig = plt.figure(figsize=(8.0, 4.0))
    left, right = fig.subfigures(1, 2)
    ax = left.subplots()
    sibling = right.subplots()
    sibling.plot([0.0, 1.0], [1.0, 0.0])
    sibling.set_title("Independent panel")
    before = _snapshot(sibling)
    size_before = fig.get_size_inches().copy()
    result = _plot(uniform_data, ax=ax, return_result=True)
    assert result.fig is fig
    assert result.axes["field"] is ax
    assert result.metadata["export_scope"] == "parent_figure"
    assert "sibling" in result.metadata["source_scope"]
    result.save(tmp_path / "composed_field", formats="png", dpi=40)
    np.testing.assert_array_equal(fig.get_size_inches(), size_before)
    _assert_untouched(sibling, before)


@pytest.fixture
def paga_data(uniform_data):
    points = np.vstack([uniform_data.obsm["X_umap"], [0.0, 0.0]])
    velocities = np.vstack([uniform_data.obsm["velocity_umap"], [1000.0, -1000.0]])
    conditions = uniform_data.obs.condition.astype(str).tolist() + ["unrelated"]
    data = _make_data(points, velocities, conditions)
    data.obs["state"] = pd.Categorical(
        np.where(points[:, 0] < -0.2, "A", np.where(points[:, 0] > 0.2, "C", "B")),
        categories=["A", "B", "C"],
    )
    data.uns["paga"] = {
        "groups": "state",
        "connectivities": sparse.csr_matrix(
            [[0.0, 0.7, 0.2], [0.7, 0.0, 0.8], [0.2, 0.8, 0.0]]
        ),
    }
    return data


def _paga(data, **kwargs):
    options = {
        "node_key": "state",
        "condition_key": "condition",
        "group0": "control",
        "group1": "treated",
        "graph_node_order": ["A", "B", "C"],
        "min_cells": 1,
        "show_velocity": True,
        "return_result": True,
        "show": False,
    }
    options.update(kwargs)
    return sg.pl.paga_shift_map(data, **options)


def _field_options():
    return {"grid_size": 11, "n_neighbors": 16, "radius": 0.75, "min_neighbors": 3}


def test_paga_default_velocity_mode_preserves_state_vectors(paga_data):
    default = _paga(paga_data)
    explicit = _paga(paga_data, velocity_mode="state")
    assert default.metadata["velocity_mode"] == "state"
    for key in ("centroids", "velocity", "vectors", "edge_table"):
        pd.testing.assert_frame_equal(default.data[key], explicit.data[key])
    assert (
        len(default.artists["vectors"]["velocity"])
        == len(explicit.artists["vectors"]["velocity"])
        > 0
    )
    assert "velocity_field" not in default.artists


@pytest.mark.parametrize("mode", ["grid", "stream"])
def test_paga_local_fields_enforce_selected_contrast_without_duplicate_state_arrows(
    paga_data, mode
):
    result = _paga(paga_data, velocity_mode=mode, velocity_options=_field_options())
    assert result.metadata["velocity_mode"] == mode
    assert result.metadata["velocity_scope"] == "selected_conditions"
    assert result.metadata["velocity_scale"] is None
    assert result.metadata["local_velocity"]
    supported = result.data["velocity_grid"].loc[lambda frame: frame.supported]
    assert not supported.empty
    np.testing.assert_allclose(
        supported[["vx", "vy"]], np.tile([1.0, 0.25], (len(supported), 1))
    )
    sources = result.data["velocity_cells"]
    assert len(sources) == paga_data.n_obs
    assert not sources.loc[sources.condition.eq("unrelated"), "eligible"].any()
    legacy = result.data["vectors"].loc[lambda frame: frame.kind.eq("velocity")]
    assert not legacy.displayed.any()
    assert legacy.reason.eq("local_field_mode").all()
    assert not result.artists["vectors"]["velocity"]
    artists = result.artists["velocity_field"]
    if mode == "grid":
        assert artists["grid_arrows"]
    else:
        assert len(artists["stream_lines"].get_segments()) > 0
        assert not result.data["velocity_paths"].empty


@pytest.mark.parametrize("forbidden", ["groups", "condition_key", "color_by"])
def test_paga_velocity_options_cannot_override_scope_or_competing_numeric_color(
    paga_data, forbidden
):
    _, ax = plt.subplots()
    before = _snapshot(ax)
    options = _field_options()
    options[forbidden] = (
        ["unrelated"]
        if forbidden == "groups"
        else "condition"
        if forbidden == "condition_key"
        else "coherence"
    )
    with pytest.raises(
        ValueError, match="(?i)(velocity_options|unsupported|unknown|allowed)"
    ):
        _paga(paga_data, ax=ax, velocity_mode="grid", velocity_options=options)
    _assert_untouched(ax, before)


def test_paga_local_mode_keeps_existing_nonfinite_coordinate_rejection(paga_data):
    paga_data.obsm["X_umap"][0] = [np.nan, 0.0]
    _, ax = plt.subplots()
    before = _snapshot(ax)
    with pytest.raises(ValueError, match="(?i)(nonfinite|coordinates)"):
        _paga(paga_data, ax=ax, velocity_mode="grid", velocity_options=_field_options())
    _assert_untouched(ax, before)


def test_paga_local_field_export_includes_sources_grid_vectors_and_paths(
    paga_data, tmp_path
):
    result = _paga(paga_data, velocity_mode="stream", velocity_options=_field_options())
    output = result.save(tmp_path / "paga_stream", formats="png", dpi=40)
    metadata = json.loads(Path(output["metadata"]).read_text())
    for key in (
        "velocity_cells",
        "velocity_grid",
        "velocity_vectors",
        "velocity_paths",
    ):
        path = Path(output["tables"][key])
        assert path.is_file()
        assert (
            metadata["tables"][key]["sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
        loaded = pd.read_csv(path)
        assert len(loaded) == len(result.data[key])
    assert metadata["plot"]["velocity_mode"] == "stream"
    assert metadata["plot"]["velocity_scale"] is None


@pytest.mark.parametrize("density", [0.05, (0.05, 1.0), (1.0, 0.05)])
@pytest.mark.parametrize("entry_point", ["standalone", "paga"])
def test_low_stream_density_rejected_before_axes_mutation(
    uniform_data, paga_data, density, entry_point
):
    _, ax = plt.subplots()
    ax.plot([0.0, 1.0], [1.0, 0.0])
    ax.set(title="Caller panel", xlim=(-3.0, 4.0), ylim=(-2.0, 5.0))
    before = _snapshot(ax)
    with pytest.raises(ValueError, match="(?i)(stream_density|density)"):
        if entry_point == "standalone":
            _plot(uniform_data, ax=ax, mode="stream", stream_density=density)
        else:
            options = _field_options()
            options["stream_density"] = density
            _paga(paga_data, ax=ax, velocity_mode="stream", velocity_options=options)
    _assert_untouched(ax, before)
