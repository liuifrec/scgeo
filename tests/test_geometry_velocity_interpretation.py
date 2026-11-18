"""Interpretation plots: no fabricated states, common velocity scales, audited focus."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import scgeo as sg


@pytest.fixture
def geometry_tables():
    shift = pd.DataFrame({
        "state": ["a", "b", "c", "d"],
        "coverage_status": ["eligible_exploratory", "eligible_exploratory", "insufficient_coverage", "eligible_exploratory"],
        "normalized_delta_norm": [2.1, 3.5, np.nan, 1.8],
        "delta_norm": [1.5, 4.0, np.nan, 2.2],
        "magnitude_ci95_low": [1.2, 3.1, np.nan, 1.7],
        "magnitude_ci95_high": [1.9, 5.0, np.nan, 2.7],
        "n_mice_control": [2, 2, 1, 2],
        "n_mice_irradiated": [2, 2, 2, 2],
    })
    align = pd.DataFrame({"leiden": ["d", "b", "c", "a"],
                          "alignment_cosine": [-0.8, 0.1, np.nan, 0.9],
                          "usable": [True, True, False, True], "rep": ["X_pca"]*4})
    return shift, align


def test_concordance_uses_state_identity_not_table_order(geometry_tables):
    shift, align = geometry_tables
    before_shift, before_align = shift.copy(deep=True), align.copy(deep=True)
    result = sg.pl.geometry_velocity_concordance(
        shift, align, analysis_rep="X_pca", return_result=True, show=False
    )
    assert result.metadata["n_eligible"] == 3
    assert result.metadata["n_not_assessed"] == 1
    table = result.data["eligible"].set_index("_state_id")
    assert table.loc["a", "alignment"] == pytest.approx(0.9)
    assert table.loc["b", "alignment"] == pytest.approx(0.1)
    assert table.loc["d", "alignment"] == pytest.approx(-0.8)
    assert not table.ci_low.notna().any()  # raw CI must not appear for normalized effect
    assert shift.equals(before_shift) and align.equals(before_align)
    plt.close(result.fig)


def test_concordance_preserves_raw_interval_exactly(geometry_tables):
    shift, align = geometry_tables
    result = sg.pl.geometry_velocity_concordance(
        shift, align, effect_metric="delta_norm", return_result=True, show=False
    )
    table = result.data["eligible"].set_index("_state_id")
    assert table.loc["a", "ci_low"] == pytest.approx(1.2)
    assert table.loc["d", "ci_high"] == pytest.approx(2.7)
    assert result.metadata["interval_columns"] == ["magnitude_ci95_low", "magnitude_ci95_high"]
    plt.close(result.fig)


def test_concordance_rejects_duplicate_ids_invalid_interval_and_cosine(geometry_tables):
    shift, align = geometry_tables
    with pytest.raises(ValueError, match="duplicate"):
        sg.pl.geometry_velocity_concordance(pd.concat([shift, shift.iloc[:1]]), align, show=False)
    bad = align.copy()
    bad.loc[0, "alignment_cosine"] = 1.5
    with pytest.raises(ValueError, match="between -1 and 1"):
        sg.pl.geometry_velocity_concordance(shift, bad, show=False)
    bad_shift = shift.copy()
    bad_shift.loc[0, "magnitude_ci95_low"] = 2.0
    with pytest.raises(ValueError, match="ordered"):
        sg.pl.geometry_velocity_concordance(bad_shift, align, effect_metric="delta_norm", show=False)


@pytest.fixture
def mouse_like_adata():
    rng = np.random.RandomState(18)
    n = 240
    labels = np.array(["reference"] * n + ["treated"] * n, dtype=object)
    xy = np.vstack([rng.normal(loc=(0, 0), scale=(0.45, 0.65), size=(n, 2)),
                    rng.normal(loc=(0.14, -0.09), scale=(0.45, 0.65), size=(n, 2))])
    vel = np.column_stack([np.ones(n*2), np.zeros(n*2)]) * 0.12
    vel[n:, 0] = -0.12
    obs = pd.DataFrame({"condition": labels,
                        "state": pd.Categorical(["A"]*(n//2)+["B"]*(n//2)+
                                                ["A"]*(n//2)+["B"]*(n//2)),
                       }, index=[f"c{i}" for i in range(2*n)])
    adata = AnnData(np.zeros((2*n, 2)), obs=obs)
    adata.obsm["X_umap"] = xy
    adata.obsm["velocity_umap"] = vel
    adata.uns["paga"] = {
        "groups": "state",
        "connectivities": sparse.csr_matrix([[0, 0.8], [0.8, 0]]),
    }
    yield adata
    plt.close("all")


def test_velocity_conditions_share_grid_and_do_not_pool_opposing_vectors(mouse_like_adata):
    adata = mouse_like_adata
    before = adata.obsm["velocity_umap"].copy()
    out = sg.pl.velocity_condition_compare(
        adata, condition_key="condition", group0="reference", group1="treated",
        grid_size=13, radius=0.65, n_neighbors=25, min_neighbors=5,
        min_coherence=0.1, return_result=True, show=False,
    )
    table = out.data["paired_grid"]
    both = table.loc[table.support_status == "both_supported"]
    assert len(both) > 0
    assert (both.direction_cosine < -0.95).all()
    assert np.array_equal(out.data["reference_grid"][["x", "y"]].to_numpy(),
                          out.data["comparison_grid"][["x", "y"]].to_numpy())
    assert np.allclose(out.data["reference_grid"].loc[
        out.data["reference_grid"].supported, "vx"], 0.12)
    assert np.allclose(out.data["comparison_grid"].loc[
        out.data["comparison_grid"].supported, "vx"], -0.12)
    assert out.metadata["shared_scale"] > 0
    np.testing.assert_array_equal(adata.obsm["velocity_umap"], before)
    plt.close(out.fig)


def test_velocity_comparison_streamlines_and_absent_velocity(mouse_like_adata):
    adata = mouse_like_adata
    out = sg.pl.velocity_condition_compare(
        adata, condition_key="condition", group0="reference", group1="treated",
        grid_size=12, radius=0.7, n_neighbors=30, min_neighbors=4,
        mode="stream", stream_density=1.0, return_result=True, show=False,
    )
    assert len(out.data["paired_grid"]) == 144
    assert out.metadata["n_both_supported"] > 0
    plt.close(out.fig)
    del adata.obsm["velocity_umap"]
    with pytest.raises(KeyError, match="velocity_umap"):
        sg.pl.velocity_condition_compare(adata, condition_key="condition",
                                         group0="reference", group1="treated", show=False)


def test_paga_focus_only_changes_display_and_keeps_original_graph(mouse_like_adata):
    adata = mouse_like_adata
    original = adata.uns["paga"]["connectivities"].copy()
    result = sg.pl.paga_shift_map(
        adata, node_key="state", condition_key="condition", group0="reference",
        group1="treated", min_cells=1, focus_states=["A"], edge_focus="within",
        label_top_n=0, return_result=True, show=False,
    )
    table = result.data["edge_table"]
    assert table.displayed.sum() == 0
    assert table.reason.str.contains("outside_focus").any()
    assert result.metadata["focus_states"] == ["A"]
    assert result.metadata["edge_focus"] == "within"
    assert np.array_equal(adata.uns["paga"]["connectivities"].toarray(), original.toarray())
    plt.close(result.fig)
    with pytest.raises(ValueError, match="focus_states"):
        sg.pl.paga_shift_map(adata, node_key="state", condition_key="condition",
                             group0="reference", group1="treated", focus_states=["missing"],
                             edge_focus="within", show=False)
