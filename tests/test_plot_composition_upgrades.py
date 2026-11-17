"""Data/artist contracts for composable plots and stored alignment semantics."""

from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.colors import to_rgba
from scipy import sparse

import scgeo as sg


@pytest.fixture
def adata():
    obs = pd.DataFrame(
        {
            "state": pd.Categorical(["A"] * 8 + ["B"] * 8),
            "condition": ["control"] * 5
            + ["treated"] * 3
            + ["control"] * 3
            + ["treated"] * 5,
            "score": np.arange(16, dtype=float),
            "scgeo_ood": np.linspace(0, 1, 16),
        },
        index=[f"c{i}" for i in range(16)],
    )
    ad = AnnData(np.zeros((16, 3)), obs=obs)
    xy = np.column_stack([np.arange(16) % 8, np.repeat([0.0, 3.0], 8)])
    xy[:, 1] += np.tile(np.linspace(0, 0.6, 8), 2)
    ad.obsm["X_umap"] = xy
    ad.obsm["velocity_umap"] = np.tile([0.04, 0.02], (16, 1))
    ad.obsm["X_pca"] = np.column_stack([xy, np.arange(16)])
    ad.obsm["velocity_pca"] = np.tile([0.04, 0.02, 0.3], (16, 1))
    ad.uns["paga"] = {"connectivities": sparse.csr_matrix([[0, 0.8], [0.8, 0]])}
    table = pd.DataFrame(
        {
            "node": ["A", "B"],
            "rep": ["X_pca", "X_pca"],
            "alignment_cosine": [0.83, -0.42],
            "usable": [True, True],
        }
    )
    ad.uns["scgeo"] = {
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
    yield ad
    plt.close("all")


def _contrast():
    return {
        "node_key": "state",
        "condition_key": "condition",
        "group0": "control",
        "group1": "treated",
    }


def test_composition_embeds_without_creating_another_figure_or_needing_embedding(
    adata, monkeypatch
):
    del adata.obsm["X_umap"]
    fig, ax = plt.subplots()
    monkeypatch.setattr(
        fig, "tight_layout", lambda *a, **k: pytest.fail("Caller owns layout")
    )
    before = list(plt.get_fignums())
    result = sg.pl.composition_drift(
        adata,
        **_contrast(),
        panels=("change",),
        axes={"change": ax},
        state_order=["B", "A"],
        return_result=True,
        show=False,
    )
    assert plt.get_fignums() == before
    assert result.fig is fig and result.axes["change"] is ax
    assert len(ax.patches) == 2
    assert result.data.state.tolist() == ["B", "A"]
    assert result.data.set_index("state").loc["A", "delta_frac"] == pytest.approx(
        3 / 8 - 5 / 8
    )
    assert result.metadata["inference_level"] == "descriptive_cell_fractions"


def test_composition_supplied_axes_and_palette(adata):
    fig, aa = plt.subplots(1, 3)
    axes = dict(zip(("embedding", "change", "proportions"), aa))
    result = sg.pl.composition_drift(
        adata,
        **_contrast(),
        axes=axes,
        condition_palette={"control": "#123456", "treated": "#abcdef"},
        return_result=True,
        show=False,
    )
    assert result.fig is fig
    assert result.artists["group0"][0].get_facecolor()[:3] == pytest.approx(
        to_rgba("#123456")[:3]
    )
    assert list(result.axes) == list(axes)


def test_composition_subfigure_support(adata):
    fig = plt.figure()
    subfig = fig.subfigures(1, 1)
    result = sg.pl.composition_drift(
        adata,
        **_contrast(),
        subfigure=subfig,
        panels=("change", "proportions"),
        return_result=True,
        show=False,
    )
    assert result.axes["change"].figure is subfig
    assert len(plt.get_fignums()) == 1


def test_gallery_contains_data_in_composition_panel(adata):
    result = sg.pl.gallery_overview(
        adata, **_contrast(), return_result=True, show=False
    )
    ax = result.axes["composition"]
    widths = [patch.get_width() for patch in ax.patches]
    assert sorted(widths) == pytest.approx([-0.25, 0.25])
    assert not any("Use sg.pl" in text.get_text() for text in ax.texts)
    assert result.data["composition"].shape[0] == 2


def test_robustness_separates_coverage_and_evaluated_pass_rate():
    frame = pd.DataFrame(
        [[0.82, np.nan, 0.91], [np.nan, np.nan, np.nan]], index=["A", "B"]
    )
    result = sg.pl.robustness_matrix(
        frame, summary="pass_rate", pass_threshold=0.7, return_result=True, show=False
    )
    assert result.data["row_summary"].loc["A"] == 1
    assert result.data["coverage"].loc["A"] == pytest.approx(2 / 3)
    assert result.data["n_evaluated"].loc["A"] == 2
    assert np.isnan(result.data["row_summary"].loc["B"])
    assert result.data["coverage"].loc["B"] == 0
    assert any(t.get_text() == "2/3" for t in result.axes["summary"].texts)
    plt.close(result.fig)


def test_robustness_historical_missing_policy_and_shared_axes():
    fig, aa = plt.subplots(1, 2)
    frame = pd.DataFrame([[0.82, np.nan, 0.91]])
    result = sg.pl.robustness_matrix(
        frame,
        summary="pass_rate",
        pass_threshold=0.7,
        missing_policy="fail",
        axes={"heatmap": aa[0], "summary": aa[1]},
        return_result=True,
        show=False,
    )
    assert result.fig is fig
    assert result.data["row_summary"].iloc[0] == pytest.approx(2 / 3)
    assert result.data["evaluated_pass_rate"].iloc[0] == 1
    assert result.metadata["missing_policy"] == "fail"
    plt.close(fig)


def test_score_alias_sampling_and_full_color_limits(adata):
    result = sg.pl.score_embedding(
        adata,
        "score",
        basis="X_umap",
        max_points=4,
        stratify_by="state",
        random_state=7,
        return_result=True,
        show=False,
    )
    other = sg.pl.score_embedding(
        adata,
        "score",
        basis="umap",
        max_points=4,
        stratify_by="state",
        random_state=7,
        return_result=True,
        show=False,
    )
    assert result.metadata["displayed_indices"] == other.metadata["displayed_indices"]
    assert result.data.displayed.sum() == 4
    assert set(result.data.loc[result.data.displayed, "stratum"]) == {"A", "B"}
    assert result.artists["scores"].norm.vmin == 0
    assert result.artists["scores"].norm.vmax == 15
    assert len(result.data) == adata.n_obs


def test_score_palette_stable_across_subset(adata):
    full = sg.pl.score_embedding(adata, "state", return_result=True, show=False)
    subset = sg.pl.score_embedding(
        adata[adata.obs.state == "B"].copy(), "state", return_result=True, show=False
    )
    assert full.metadata["palette"]["B"] == subset.metadata["palette"]["B"]
    assert full.axes["embedding"].get_legend() is not None


def test_score_rejects_inadequate_sampling_budget(adata):
    with pytest.raises(ValueError, match="preserve"):
        sg.pl.score_embedding(
            adata, "score", stratify_by="state", max_points=1, show=False
        )


def test_stored_alignment_fixed_under_umap_change(adata):
    original = copy.deepcopy(adata.uns["scgeo"]["representation_stability"])
    first = sg.pl.velocity_shift_alignment(
        adata,
        **_contrast(),
        mode="stored",
        analysis_rep="X_pca",
        min_cells=1,
        return_result=True,
        show=False,
    )
    adata.obsm["X_umap"] = adata.obsm["X_umap"] @ np.array([[2, 1], [-3, 4]])
    second = sg.pl.velocity_shift_alignment(
        adata,
        **_contrast(),
        mode="stored",
        analysis_rep="X_pca",
        min_cells=1,
        return_result=True,
        show=False,
    )
    pd.testing.assert_frame_equal(
        first.data["stored_alignment"], second.data["stored_alignment"]
    )
    assert first.metadata["quantitative_values_recomputed"] is False
    assert (
        first.axes["overlay"].get_ylim()[1] > 4.0
    )  # Include projected-arrow endpoints.
    assert not first.data["display_vectors"].equals(second.data["display_vectors"])
    pd.testing.assert_frame_equal(
        original["per_rep_state"],
        adata.uns["scgeo"]["representation_stability"]["per_rep_state"],
    )
    np.testing.assert_allclose(
        [
            collection.get_offsets()[0, 0]
            for collection in first.axes["alignment"].collections
        ],
        [0.83, -0.42],
    )


@pytest.mark.parametrize("incompatible", [False, True])
def test_stored_alignment_without_compatible_display_velocity(adata, incompatible):
    if not incompatible:
        del adata.obsm["velocity_umap"]
    result = sg.pl.velocity_shift_alignment(
        adata,
        **_contrast(),
        mode="stored",
        analysis_rep="pca",
        velocity_basis="pca" if incompatible else None,
        min_cells=1,
        return_result=True,
        show=False,
    )
    assert result.metadata["projected_velocity_key"] is None
    assert result.metadata["analysis_rep"] == "X_pca"
    assert result.warnings
    assert list(result.data["stored_alignment"].alignment_cosine) == [0.83, -0.42]
    assert "vx" not in result.data["display_vectors"]


def test_stored_alignment_requires_rep_selection_for_multiple_reps(adata):
    payload = adata.uns["scgeo"]["representation_stability"]
    other = payload["per_rep_state"].assign(rep="X_pca20", alignment_cosine=0.1)
    payload["per_rep_state"] = pd.concat(
        [payload["per_rep_state"], other], ignore_index=True
    )
    with pytest.raises(ValueError, match="analysis_rep"):
        sg.pl.velocity_shift_alignment(adata, **_contrast(), mode="stored", show=False)
    selected = sg.pl.velocity_shift_alignment(
        adata,
        **_contrast(),
        mode="stored",
        analysis_rep="X_pca20",
        return_result=True,
        show=False,
    )
    assert selected.data["stored_alignment"].alignment_cosine.eq(0.1).all()


def test_stored_alignment_validates_contrast_and_metadata(adata):
    payload = adata.uns["scgeo"]["representation_stability"]
    payload["params"]["group1"] = "other"
    with pytest.raises(ValueError, match="does not match"):
        sg.pl.velocity_shift_alignment(adata, **_contrast(), mode="stored", show=False)
    plain = payload["per_rep_state"].drop(columns="rep")
    with pytest.raises(ValueError, match="analysis_rep"):
        sg.pl.velocity_shift_alignment(
            adata, **_contrast(), mode="stored", alignment_table=plain, show=False
        )


def test_stored_alignment_preserves_unavailable_states_and_accepts_axes(adata):
    payload = adata.uns["scgeo"]["representation_stability"]
    payload["per_rep_state"].loc[1, "alignment_cosine"] = np.nan
    fig, aa = plt.subplots(1, 2)
    result = sg.pl.velocity_shift_alignment(
        adata,
        **_contrast(),
        mode="stored",
        min_cells=1,
        axes={"overlay": aa[0], "alignment": aa[1]},
        state_order=["B", "A"],
        return_result=True,
        show=False,
    )
    assert result.fig is fig
    assert result.data["stored_alignment"].state.tolist() == ["B", "A"]
    assert any(t.get_text() == "unavailable" for t in aa[1].texts)


def test_velocity_legend_uses_configured_colors(adata):
    result = sg.pl.velocity_shift_alignment(
        adata,
        **_contrast(),
        min_cells=1,
        shift_color="red",
        velocity_color="green",
        return_result=True,
        show=False,
    )
    legend = result.axes["overlay"].get_legend()
    assert [line.get_color() for line in legend.get_lines()] == ["red", "green"]
    assert "display" in result.axes["overlay"].get_title().lower()


def test_score_sampling_retains_required_cells(adata):
    result = sg.pl.score_embedding(
        adata,
        "score",
        stratify_by="state",
        max_points=3,
        required_indices=[0, 15],
        return_result=True,
        show=False,
    )
    assert {0, 15} <= set(result.metadata["displayed_indices"])
    assert result.metadata["required_indices"] == [0, 15]
    with pytest.raises(ValueError, match="out-of-range"):
        sg.pl.score_embedding(adata, "score", required_indices=[16], show=False)


def test_stored_alignment_rejects_conflicting_representation_label(adata):
    table = adata.uns["scgeo"]["representation_stability"]["per_rep_state"].drop(
        columns="rep"
    )
    table.attrs["params"] = {"basis": "pca"}
    with pytest.raises(ValueError, match="representation metadata"):
        sg.pl.velocity_shift_alignment(
            adata,
            **_contrast(),
            mode="stored",
            alignment_table=table,
            analysis_rep="umap",
            show=False,
        )


def test_stored_alignment_plain_table_declares_unverified_contrast(adata):
    table = adata.uns["scgeo"]["representation_stability"]["per_rep_state"].drop(
        columns="rep"
    )
    table["alignment_cosine"] = pd.Series([pd.NA, -0.42], dtype="Float64")
    result = sg.pl.velocity_shift_alignment(
        adata,
        **_contrast(),
        mode="stored",
        alignment_table=table,
        analysis_rep="X_pca",
        return_result=True,
        show=False,
    )
    assert any("metadata is incomplete" in note for note in result.warnings)
    assert any(
        text.get_text() == "unavailable" for text in result.axes["alignment"].texts
    )


def test_stored_overlay_excludes_unrelated_conditions(adata):
    adata.obs.loc[adata.obs.index[-4:], "condition"] = "unrelated"
    adata.obsm["velocity_umap"][-4:] = [0, 100]
    result = sg.pl.velocity_shift_alignment(
        adata,
        **_contrast(),
        mode="stored",
        min_cells=1,
        return_result=True,
        show=False,
    )
    vectors = result.data["display_vectors"].set_index("state")
    np.testing.assert_allclose(
        vectors.loc["B", ["vx", "vy"]].to_numpy(dtype=float), [0.04, 0.02]
    )
    assert result.metadata["projected_velocity_scope"] == "selected_conditions"
    assert result.metadata["projected_velocity_conditions"] == ["control", "treated"]
    assert result.data["stored_alignment"].alignment_cosine.tolist() == [0.83, -0.42]


def test_stored_alignment_exports_exact_background_cells(adata, tmp_path):
    adata.obsm["X_umap"][1] = [np.nan, 0]
    result = sg.pl.velocity_shift_alignment(
        adata,
        **_contrast(),
        mode="stored",
        min_cells=1,
        return_result=True,
        show=False,
    )
    cells = result.data["display_cells"]
    assert cells.obs_position.tolist() == [0] + list(range(2, 16))
    assert cells.cell.tolist() == adata.obs_names[cells.obs_position].tolist()
    assert (
        cells.state.tolist()
        == adata.obs.iloc[cells.obs_position].state.astype(str).tolist()
    )
    np.testing.assert_allclose(
        result.artists["background"].get_offsets(), cells[["x", "y"]].to_numpy()
    )
    manifest = result.save(tmp_path / "stored", formats=("png",), dpi=60)
    reloaded = pd.read_csv(manifest["tables"]["display_cells"])
    np.testing.assert_allclose(reloaded[["x", "y"]], cells[["x", "y"]])
    assert (
        result.metadata["background_scope"]
        == "all_conditions_finite_display_coordinates"
    )
    assert result.metadata["projected_velocity_scope"] == "selected_conditions"
