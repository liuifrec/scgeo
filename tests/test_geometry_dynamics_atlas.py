"""General-purpose atlas contracts: AnnData portability and honest evidence.

All examples are simulated, with deliberate missing conditions, nonfinite
coordinates, an unrelated population, and display-only projected vectors.
"""

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
from scipy import sparse

import scgeo as sg


@pytest.fixture(autouse=True)
def figures_closed():
    yield
    plt.close("all")


@pytest.fixture
def data():
    rng = np.random.default_rng(731)
    positions, vel, groups, conditions, batches = [], [], [], [], []
    # 3 cell states * 2 conditions * 50 cells per condition; reference graph
    # has actual finite, symmetric, undirected connectivity.
    for state, base in (("A", (0., 0.)), ("B", (3., 1.)), ("C", (1., 4.))):
        for condition in ("control", "treated"):
            for i in range(50):
                p = rng.normal(0., .30, 2) + base
                if condition == "treated":
                    p += (.4, .2)
                positions.append(p)
                v = (.07, .025) if state != "B" else (-.05, .04)
                vel.append(np.array(v) + rng.normal(0., .006, 2))
                groups.append(state)
                conditions.append(condition)
                batches.append(f"{condition}_{i % 5}")
    obs = pd.DataFrame({
        "state": pd.Categorical(groups, categories=["C", "A", "B"], ordered=True),
        "condition": conditions,
        "sample": batches,
        "score": np.linspace(0.0, 1.0, len(groups)),
    }, index=[f"cell{i:04d}" for i in range(len(groups))])
    adata = AnnData(X=np.zeros((len(obs), 2)), obs=obs)
    adata.obsm["X_umap"] = np.asarray(positions)
    adata.obsm["velocity_umap"] = np.asarray(vel)
    adata.uns["paga"] = {
        "groups": "state",
        "connectivities": sparse.csr_matrix([
            [0, .9, .5], [.9, 0, .2], [.5, .2, 0],
        ]),
    }
    return adata


@pytest.fixture
def evidence():
    result = pd.DataFrame({
        "state": ["B", "A", "C"],
        "delta_norm": [.8, .4, np.nan],
        "normalized_delta_norm": [.6, .2, np.nan],
        "magnitude_ci95_low": [.6, .3, np.nan],
        "magnitude_ci95_high": [.9, .55, np.nan],
        "effect_status": ["assessed", "assessed", "insufficient_coverage"],
    })
    result.attrs["provenance"] = {
        "group0": "control", "group1": "treated", "rep": "X_pca",
        "node_key": "state",
    }
    return result


def _common(data, **kw):
    kwargs = dict(color="state", node_key="state", condition_key="condition",
                  group0="control", group1="treated", show=False,
                  velocity_options={"grid_size": 12, "n_neighbors": 16,
                                    "min_neighbors": 5, "min_coherence": .05},
                  paga_options={"min_cells": 5})
    kwargs.update(kw)
    return sg.pl.geometry_dynamics_atlas(data, **kwargs)


def test_minimal_ordinary_anndata_works_without_velocity_or_scgeo():
    adata = AnnData(X=np.zeros((12, 1)), obs=pd.DataFrame({
        "type": ["T", "B", "Mono"] * 4,
    }, index=[f"x{i}" for i in range(12)]))
    adata.obsm["X_umap"] = np.random.default_rng(1).normal(size=(12, 2))
    result = sg.pl.geometry_dynamics_atlas(adata, color="type", show=False)
    assert isinstance(result, sg.pl.PlotResult)
    assert result.metadata["panels"] == ["states"]
    assert set(result.metadata["omitted_panels"]) == {"velocity", "paga", "evidence"}
    assert len(result.data["cells"]) == 12
    assert result.data["cells"].displayed.all()
    assert result.axes["states"].get_aspect() == 1


def test_composed_atlas_reuses_axes_without_parent_layout_changes(data, evidence, tmp_path):
    reference = _common(data, report=evidence, velocity_mode="grid",
                        paga_options={"min_cells": 5, "label": False})
    fig = plt.figure(figsize=(14, 7))
    left, right = fig.subfigures(1, 2)
    sibling = right.subplots()
    sibling.set_title("Caller sibling")
    axs = left.subplots(2, 2)
    names = ("states", "velocity", "paga", "evidence")
    mapping = dict(zip(names, axs.flat))
    positions = [ax.get_position(original=True).bounds for ax in axs.flat]
    fig.suptitle("Caller title")
    result = _common(data, report=evidence, velocity_mode="grid", axes=mapping,
                     figsize=(1, 1), title="Ignored on supplied canvas",
                     paga_options={"min_cells": 5, "label": False})
    assert result.fig is fig and result.axes == mapping
    assert len(fig.axes) == 5
    assert fig._suptitle.get_text() == "Caller title"
    assert sibling.get_title() == "Caller sibling"
    np.testing.assert_array_equal(fig.get_size_inches(), [14, 7])
    np.testing.assert_allclose([ax.get_position(original=True).bounds for ax in axs.flat], positions)
    for name in reference.data:
        pd.testing.assert_frame_equal(reference.data[name], result.data[name])
    output = result.save(tmp_path / "atlas", formats=("svg",), dpi=60)
    assert json.loads(Path(output["metadata"]).read_text())["plot"]["export_scope"] == "parent_figure"


def test_atlas_numeric_colorbar_and_axes_validation(data):
    fig, axs = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="colorbar"):
        _common(data, color="score", panels=["states"], axes={"states": axs[0]})
    assert not axs[0].collections
    result = _common(data, color="score", panels=["states"],
                     axes={"states": axs[0], "colorbar": axs[1]})
    assert result.artists["states_colorbar"].ax is axs[1]
    assert len(fig.axes) == 2
    with pytest.raises(ValueError, match="distinct Axes"):
        _common(data, panels=["states", "velocity"], axes=dict.fromkeys(["states", "velocity"], axs[0]))
    with pytest.raises(TypeError, match="mapping"):
        _common(data, panels=["states"], axes=axs)
    with pytest.raises(ValueError, match="missing panels"):
        _common(data, panels=["states", "velocity"], axes={"states": axs[0]})


def test_auto_selects_four_panels_and_preserves_source_tables(data, evidence):
    adata_copy = data.copy()
    report_copy = evidence.copy(deep=True)
    result = _common(data, report=evidence, velocity_mode="grid", sample_key="sample",
                     highlight_states=["B"], max_points=100)
    assert result.metadata["panels"] == ["states", "velocity", "paga", "evidence"]
    assert set(result.axes) == set(result.metadata["panels"])
    assert len(result.data["cells"]) == data.n_obs
    assert result.data["cells"].displayed.sum() == 100
    assert result.metadata["unique_samples_by_condition"] == {"control": 5, "treated": 5}
    assert result.metadata["drawn_state_cells"] == 100
    assert "velocity_grid" in result.data
    assert "paga_edge_table" in result.data
    assert "evidence_all" in result.data
    assert set(result.data["evidence_all"]._state_id) == {"A", "B", "C"}
    assert len(result.data["paga_edge_table"]) == 3
    pd.testing.assert_frame_equal(evidence, report_copy)
    np.testing.assert_array_equal(data.obsm["velocity_umap"], adata_copy.obsm["velocity_umap"])
    np.testing.assert_array_equal(data.obsm["X_umap"], adata_copy.obsm["X_umap"])
    for k in ("states", "velocity", "paga"):
        np.testing.assert_allclose(result.axes[k].get_xlim(), result.axes["states"].get_xlim())
        np.testing.assert_allclose(result.axes[k].get_ylim(), result.axes["states"].get_ylim())
        assert result.axes[k].get_aspect() == 1


def test_auto_observes_missing_velocity_but_retains_geometry(data, evidence):
    del data.obsm["velocity_umap"]
    result = _common(data, report=evidence)
    assert result.metadata["panels"] == ["states", "paga", "evidence"]
    assert "velocity" in result.metadata["omitted_panels"]
    assert "velocity_grid" not in result.data


def test_explicit_unavailable_panel_fails_before_creating_figure(data):
    data.obsm.pop("velocity_umap")
    n0 = len(plt.get_fignums())
    with pytest.raises(ValueError, match="velocity.*unavailable|unavailable:.*velocity"):
        _common(data, panels=("states", "velocity"))
    assert len(plt.get_fignums()) == n0


def test_nonmatching_velocity_shape_rejected(data):
    data.obsm["velocity_umap"] = np.zeros((data.n_obs, 3))
    with pytest.raises(ValueError, match="shapes differ"):
        _common(data)


def test_no_conditional_paga_without_contrast_and_strict_error(data):
    result = sg.pl.geometry_dynamics_atlas(data, color="state", show=False)
    assert result.metadata["panels"] == ["states", "velocity"]
    with pytest.raises(ValueError, match="paga"):
        sg.pl.geometry_dynamics_atlas(data, color="state", panels=["geometry"], show=False)


def test_inferred_two_group_contrast_is_stable(data):
    # The group order is lexicographic, not the first cell or category order.
    result = sg.pl.geometry_dynamics_atlas(
        data, color="state", condition_key="condition", show=False,
        panels=["paga"], paga_options={"min_cells": 5})
    assert result.metadata["group0"] == "control"
    assert result.metadata["group1"] == "treated"
    assert result.metadata["condition_choice"] == "inferred_two_conditions"


def test_three_groups_do_not_get_silent_contrast(data):
    data.obs.loc[data.obs.index[-5:], "condition"] = "other"
    result = sg.pl.geometry_dynamics_atlas(data, node_key="state", condition_key="condition",
                                            show=False)
    assert "paga" not in result.axes
    assert result.metadata["condition_choice"] == "none"


def test_report_missing_and_duplicate_states_rejected(data, evidence):
    new = evidence.copy(deep=True)
    new.loc[0, "state"] = "not-in-adata"
    with pytest.raises(ValueError, match="not present"):
        _common(data, report=new)
    new = evidence.copy(deep=True)
    new.loc[0, "state"] = "A"
    with pytest.raises(ValueError, match="duplicate"):
        _common(data, report=new)


def test_provenance_disagreement_rejected(data, evidence):
    new = evidence.copy(deep=True)
    new.attrs["provenance"] = dict(new.attrs["provenance"], group0="treated")
    with pytest.raises(ValueError, match="disagrees"):
        _common(data, report=new)


def test_normalized_effect_does_not_use_raw_confidence_intervals(data, evidence):
    result = _common(data, report=evidence, effect_metric="normalized_delta_norm")
    assert result.data["evidence_all"]["ci_low"].isna().all()
    assert result.metadata["panel_provenance"]["evidence"]["metric"] == "normalized_delta_norm"


def test_missing_effect_remains_unavailable_not_zero(data, evidence):
    result = _common(data, report=evidence, panels=("states", "evidence"))
    c = result.data["evidence_all"].set_index("_state_id").loc["C"]
    assert not c.assessed
    assert pd.isna(c.effect)
    assert result.metadata["panel_provenance"]["evidence"]["assessed_displayed"] == 2


def test_filter_options_block_shared_coordinate_override(data):
    with pytest.raises(ValueError, match="atlas-managed"):
        _common(data, velocity_options={"ax": plt.subplots()[1]})
    with pytest.raises(ValueError, match="atlas-managed"):
        _common(data, paga_options={"basis": "pca"})


def test_scanpy_categorical_palette_is_reused_across_panels(data):
    data.uns["state_colors"] = ["#3474A6", "#D7834B", "#419C91"]
    result = _common(data, panels=("states", "paga"))
    palette = result.metadata["panel_provenance"]["states"]["palette"]
    assert palette == {"C": "#3474A6", "A": "#D7834B", "B": "#419C91"}
    assert result.metadata["panel_provenance"]["states"]["palette_origin"] == "adata.uns['state_colors']"
    paga = result.metadata["panel_provenance"]["paga"]["node_palette"]
    assert all(paga[key] == value for key, value in palette.items())


def test_numeric_obs_coloring_and_missing_categories(data):
    data.obs["state2"] = data.obs.state.astype(object)
    data.obs.loc[data.obs.index[:5], "state2"] = None
    data.obs.loc[data.obs.index[5:7], "state2"] = "NA"
    # Missing labels are distinct from the literal 'NA' category.
    result = sg.pl.geometry_dynamics_atlas(data, color="state2", panels=("states",), show=False)
    palette = result.metadata["panel_provenance"]["states"]["palette"]
    assert "NA" in palette and "(missing)" in palette
    assert palette["NA"] != palette["(missing)"]
    numeric = sg.pl.geometry_dynamics_atlas(data, color="score", panels=("states",), show=False)
    assert numeric.metadata["panel_provenance"]["states"]["color_type"] == "numeric"


def test_svg_pdf_png_source_export_is_consistent(tmp_path, data, evidence):
    result = _common(data, report=evidence, panels=("states", "evidence"))
    exported = result.save(tmp_path / "atlas", formats=("png", "svg", "pdf"), dpi=110)
    assert {"png", "svg", "pdf"} <= set(exported["figures"])
    for path in exported["figures"].values():
        assert Path(path).exists() and Path(path).stat().st_size > 0
    assert any("cells" in name for name in exported["tables"])
    manifest = json.loads(Path(exported["metadata"]).read_text())
    assert manifest["plot"]["plot"] == "geometry_dynamics_atlas"


def test_draw_only_downsampling_does_not_reestimate_velocity(data):
    small = _common(data, panels=("states", "velocity"), max_points=30)
    large = _common(data, panels=("states", "velocity"), max_points=270)
    assert small.data["cells"].displayed.sum() == 30
    assert large.data["cells"].displayed.sum() == 270
    pd.testing.assert_frame_equal(small.data["velocity_grid"], large.data["velocity_grid"])


def test_supplied_focus_state_filters_display_only(data):
    basic = _common(data, panels=("paga",))
    focused = _common(data, panels=("paga",), highlight_states=["A"])
    assert len(basic.data["paga_edge_table"]) == len(focused.data["paga_edge_table"])
    assert int(focused.data["paga_edge_table"].displayed.sum()) <= int(basic.data["paga_edge_table"].displayed.sum())
    assert focused.metadata["panel_provenance"]["paga"]["edge_focus"] == "incident"


def test_wrong_report_scale_interval_rejected(data, evidence):
    report = evidence.copy(deep=True)
    report.loc[0, "magnitude_ci95_low"] = 2
    with pytest.raises(ValueError, match="increasing"):
        _common(data, report=report, effect_metric="delta_norm")
