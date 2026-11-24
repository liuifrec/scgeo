"""Scientific display invariants for geometric attribution curves and maps."""

from __future__ import annotations

from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from scgeo.pl._result import PlotResult
from scgeo.pl._shift_explanation import shift_curves, shift_embedding


@pytest.fixture
def explained():
    # Deliberate sample-size imbalance: sample B must have equal influence to A.
    sizes = [120, 12, 60, 20]
    levels = [0.0, 10.0, 4.0, 14.0]
    scores = np.concatenate(
        [np.linspace(0, 1, n) + (i >= 2) * 0.1 for i, n in enumerate(sizes)]
    )
    expression = np.column_stack([np.repeat(levels, sizes), scores])
    obs = pd.DataFrame(
        {
            "sample": np.repeat(["A", "B", "C", "D"], sizes),
            "condition": np.repeat(["control", "control", "treated", "treated"], sizes),
            "state": "State A",
        },
        index=[f"cell{i}" for i in range(len(scores))],
    )
    adata = AnnData(expression, obs=obs, var=pd.DataFrame(index=["G1", "G2"]))
    adata.obsm["X_display"] = np.column_stack([scores, np.sin(scores * 5)])
    cells = pd.DataFrame(
        {
            "obs_position": np.arange(adata.n_obs),
            "obs_name": adata.obs_names,
            "state": "State A",
            "condition": obs.condition.to_numpy(),
            "sample": obs["sample"].to_numpy(),
            "shift_score": scores - 0.5,
            "eligible": True,
        }
    )
    genes = pd.DataFrame(
        {
            "state": ["State A"] * 2,
            "gene": ["G1", "G2"],
            "expression_center0": [5.0, 0.5],
            "expression_center1": [9.0, 0.6],
            "expression_delta": [4.0, 0.1],
            "axis_loading": [0.0, 1.0],
            "contribution": [0.0, 0.1],
            "in_representation": [False, True],
        }
    )
    positions0 = np.flatnonzero(obs.condition.eq("control"))
    positions1 = np.flatnonzero(obs.condition.eq("treated"))
    entry = {
        "state": "State A",
        "center0": np.array([0.5]),
        "center1": np.array([0.6]),
        "delta": np.array([0.1]),
        "direction": np.array([1.0]),
        "status": "assessed",
        "obs_positions0": positions0,
        "obs_positions1": positions1,
        "center_weights0": np.repeat([0.5 / sizes[0], 0.5 / sizes[1]], sizes[:2]),
        "center_weights1": np.repeat([0.5 / sizes[2], 0.5 / sizes[3]], sizes[2:]),
    }
    adata.uns["scgeo"] = {
        "explain_shift": {
            "params": {
                "rep": "X_pca",
                "by": "state",
                "sample_key": "sample",
                "condition_key": "condition",
                "group0": "control",
                "group1": "treated",
                "layer": None,
            },
            "cells": cells,
            "genes": genes,
            "state_results": {"state_0": entry},
            "module_membership": pd.DataFrame(
                {
                    "program": ["response", "response"],
                    "gene": ["G1", "G2"],
                    "weight": [0.5, 0.5],
                }
            ),
        }
    }
    yield adata
    plt.close("all")


def test_curves_balance_samples_and_band_is_range_not_ci(explained):
    result = shift_curves(
        explained,
        ["G1"],
        panels="curve",
        bandwidth=0.3,
        min_cells=2,
        return_result=True,
        show=False,
    )
    assert isinstance(result, PlotResult)
    table = result.data["curves"]
    central = table.loc[(table.shift_score > -0.2) & (table.shift_score < 0.2)]
    control = central.loc[central.condition.eq("control")]
    treated = central.loc[central.condition.eq("treated")]
    np.testing.assert_allclose(control.expression, 5.0)
    np.testing.assert_allclose(treated.expression, 9.0)
    np.testing.assert_allclose(control.lower, 0.0)
    np.testing.assert_allclose(control.upper, 10.0)
    assert control.n_samples.eq(2).all()
    assert result.metadata["interval_is_confidence_interval"] is False
    assert result.metadata["uncertainty"] == "sample_range"
    assert "Geometric shift coordinate" in result.axes["gene:G1:curve"].get_xlabel()


def test_one_sample_has_no_fabricated_uncertainty(explained):
    bundle = explained.uns["scgeo"]["explain_shift"]
    bundle["cells"] = (
        bundle["cells"].loc[bundle["cells"]["sample"].isin(["A", "C"])].copy()
    )
    result = shift_curves(
        explained, ["G1"], panels="curve", min_cells=2, return_result=True, show=False
    )
    table = result.data["curves"]
    assert table.lower.isna().all() and table.upper.isna().all()
    assert not table.n_samples.gt(1).any()
    assert any("withheld" in message for message in result.warnings)


def test_no_sample_key_pooled_curves_have_no_sample_band(explained):
    explained.uns["scgeo"]["explain_shift"]["params"]["sample_key"] = None
    result = shift_curves(
        explained, ["G1"], panels="curve", min_cells=2, return_result=True, show=False
    )
    assert result.data["curves"].lower.isna().all()
    assert result.data["curves"].n_samples.eq(0).all()
    assert "pooled_cells_only" in set(result.data["curves"].support)


def test_display_subsampling_never_changes_curves_or_contributions(explained):
    a = shift_curves(
        explained,
        ["G2"],
        basis="display",
        max_points=20,
        min_cells=2,
        return_result=True,
        show=False,
    )
    b = shift_curves(
        explained,
        ["G2"],
        basis="display",
        max_points=None,
        min_cells=2,
        return_result=True,
        show=False,
    )
    pd.testing.assert_frame_equal(a.data["curves"], b.data["curves"])
    pd.testing.assert_frame_equal(a.data["sample_curves"], b.data["sample_curves"])
    np.testing.assert_allclose(
        a.data["cells"].contribution, b.data["cells"].contribution
    )
    assert a.data["cells"].displayed.sum() == 20
    assert b.data["cells"].displayed.sum() == explained.n_obs


def test_curves_do_not_extrapolate_or_bridge_unsupported_gap(explained):
    cells = explained.uns["scgeo"]["explain_shift"]["cells"]
    x = cells.shift_score.to_numpy()
    cells["shift_score"] = np.where(x < 0.05, x - 1.0, x + 1.0)
    result = shift_curves(
        explained,
        ["G2"],
        panels="curve",
        bandwidth=0.05,
        min_cells=2,
        grid_size=100,
        return_result=True,
        show=False,
    )
    table = result.data["curves"]
    gap = table.loc[table.shift_score.between(-0.5, 0.5)]
    assert len(gap) > 0 and gap.expression.isna().all()
    for (condition, sample), frame in result.data["sample_curves"].groupby(
        ["condition", "sample"]
    ):
        observed = cells.loc[
            cells.condition.eq(condition) & cells["sample"].eq(sample), "shift_score"
        ]
        supported = frame.loc[frame.supported, "shift_score"]
        assert supported.between(observed.min(), observed.max()).all()


def test_program_mean_expression_and_sum_contributions_with_sparse_layer(explained):
    explained.layers["source"] = sparse.csr_matrix(explained.X)
    explained.uns["scgeo"]["explain_shift"]["params"]["layer"] = "source"
    result = shift_curves(
        explained,
        program_names="response",
        basis="display",
        min_cells=2,
        return_result=True,
        show=False,
    )
    source = result.data["cells"].sort_values("obs_position")
    np.testing.assert_allclose(source.expression, explained.X.mean(axis=1))
    np.testing.assert_allclose(source.contribution, explained.X[:, 1] - 0.5)
    assert result.data["features"].iloc[0].n_attributed_members == 1


def test_stale_observation_order_and_missing_gene_fail_before_artists(explained):
    _fig, axis = plt.subplots()
    axis.set_title("Keep my title")
    before_artists = len(axis.get_children())
    with pytest.raises(KeyError, match="not present"):
        shift_curves(explained, ["missing"], panels="curve", ax=axis, show=False)
    assert len(axis.get_children()) == before_artists
    broken = explained.copy()
    broken.obs_names = list(reversed(broken.obs_names))
    with pytest.raises(ValueError, match="no longer match"):
        shift_curves(broken, ["G1"], panels="curve", ax=axis, show=False)
    assert len(axis.get_children()) == before_artists


def test_supplied_axes_are_composable_and_input_is_unchanged(explained):
    before = deepcopy(explained.uns["scgeo"]["explain_shift"])
    original_x = explained.X.copy()
    fig, array = plt.subplots(1, 3, figsize=(8, 3))
    fig.suptitle("Existing canvas")
    axes = {
        f"gene:G2:{panel}": axis
        for panel, axis in zip(("curve", "expression", "contribution"), array)
    }
    result = shift_curves(
        explained,
        ["G2"],
        basis="display",
        axes=axes,
        show_colorbar=False,
        title="Do not replace",
        return_result=True,
        show=False,
    )
    assert result.fig is fig and len(fig.axes) == 3
    assert fig._suptitle.get_text() == "Existing canvas"
    np.testing.assert_allclose(fig.get_size_inches(), [8, 3])
    np.testing.assert_array_equal(explained.X, original_x)
    pd.testing.assert_frame_equal(
        explained.uns["scgeo"]["explain_shift"]["cells"], before["cells"]
    )
    pd.testing.assert_frame_equal(
        explained.uns["scgeo"]["explain_shift"]["genes"], before["genes"]
    )


def test_missing_supplied_colorbars_fail_without_drawing(explained):
    _fig, array = plt.subplots(1, 3)
    axes = {
        f"gene:G2:{panel}": axis
        for panel, axis in zip(("curve", "expression", "contribution"), array)
    }
    before = [len(axis.get_children()) for axis in array]
    with pytest.raises(ValueError, match="missing panels"):
        shift_curves(explained, ["G2"], basis="display", axes=axes, show=False)
    assert before == [len(axis.get_children()) for axis in array]


def test_invalid_or_zero_scores_do_not_create_a_curve(explained):
    before = plt.get_fignums()
    explained.uns["scgeo"]["explain_shift"]["cells"]["shift_score"] = np.nan
    with pytest.raises(ValueError, match="no supported"):
        shift_curves(explained, ["G1"], panels="curve", show=False)
    assert plt.get_fignums() == before


def test_embedding_uses_stored_weights_and_labels_arcs_schematic(explained):
    result = shift_embedding(
        explained, basis="display", max_points=15, return_result=True, show=False
    )
    entry = explained.uns["scgeo"]["explain_shift"]["state_results"]["state_0"]
    actual = result.data["centers"].iloc[0]
    for suffix in ("0", "1"):
        expected = (
            entry[f"center_weights{suffix}"]
            @ explained.obsm["X_display"][entry[f"obs_positions{suffix}"]]
        )
        np.testing.assert_allclose(
            [actual[f"x{suffix}"], actual[f"y{suffix}"]], expected
        )
    assert actual.drawn
    assert result.metadata["not_a_trajectory"] is True
    assert len(result.artists["arrow:State A"].get_path().vertices) > 2
    assert result.data["cells"].displayed.sum() == 15


def test_embedding_zero_analysis_shift_has_no_arc_even_if_display_differs(explained):
    explained.uns["scgeo"]["explain_shift"]["state_results"]["state_0"]["status"] = (
        "zero_shift"
    )
    result = shift_embedding(explained, basis="display", return_result=True, show=False)
    assert not result.data["centers"].drawn.any()
    assert "arrow:State A" not in result.artists


def test_embedding_invalid_weights_and_state_fail_before_drawing(explained):
    _fig, axis = plt.subplots()
    before = len(axis.get_children())
    entry = explained.uns["scgeo"]["explain_shift"]["state_results"]["state_0"]
    entry["center_weights0"] = entry["center_weights0"] * 2
    with pytest.raises(ValueError, match="normalized"):
        shift_embedding(explained, basis="display", ax=axis, show=False)
    assert len(axis.get_children()) == before
    with pytest.raises(ValueError, match="Unknown states"):
        shift_embedding(
            explained, basis="display", states="missing", ax=axis, show=False
        )
    assert len(axis.get_children()) == before


def test_return_defaults_and_source_export(explained, tmp_path):
    axis = shift_curves(explained, ["G2"], panels="curve", show=False)
    assert isinstance(axis, matplotlib.axes.Axes)
    result = shift_curves(
        explained, ["G2"], panels="curve", return_result=True, show=False
    )
    paths = result.save(tmp_path / "curves", formats=("png",))
    exported = pd.read_csv(paths["tables"]["curves"])
    assert len(exported) == len(result.data["curves"])
    assert set(exported.support) == set(result.data["curves"].support)


def test_core_to_plot_integration_and_balanced_expression_change_is_rejected(explained):
    import scgeo as sg

    explained.varm["PCs"] = np.array([[0.0], [1.0]])
    explained.obsm["X_pca"] = explained.X @ explained.varm["PCs"]
    sg.tl.robust_shift(
        explained,
        rep="X_pca",
        by="state",
        condition_key="condition",
        group0="control",
        group1="treated",
        sample_key="sample",
        center="mean",
        n_boot=0,
    )
    sg.tl.explain_shift(explained, n_boot=0)
    curves = shift_curves(
        explained, ["G2"], basis="display", return_result=True, show=False
    )
    embedding = shift_embedding(
        explained, basis="display", return_result=True, show=False
    )
    assert not curves.data["curves"].expression.isna().all()
    assert embedding.data["centers"].drawn.any()
    # An orthogonal expression change preserves the mean and entire analysis
    # representation. Source integrity must still detect stale gene curves.
    explained.X[0, 0] += 1
    explained.X[1, 0] -= 1
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="changed|stale|signature"):
        shift_curves(explained, ["G1"], panels="curve", show=False)
    assert plt.get_fignums() == before


def test_changed_sample_annotation_is_rejected_before_artists(explained):
    explained.obs.loc[explained.obs_names[0], "sample"] = "changed"
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="sample labels no longer match"):
        shift_curves(explained, ["G2"], panels="curve", show=False)
    assert plt.get_fignums() == before
