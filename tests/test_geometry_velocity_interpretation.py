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
    shift = pd.DataFrame(
        {
            "state": ["a", "b", "c", "d"],
            "coverage_status": [
                "eligible_exploratory",
                "eligible_exploratory",
                "insufficient_coverage",
                "eligible_exploratory",
            ],
            "normalized_delta_norm": [2.1, 3.5, np.nan, 1.8],
            "delta_norm": [1.5, 4.0, np.nan, 2.2],
            "magnitude_ci95_low": [1.2, 3.1, np.nan, 1.7],
            "magnitude_ci95_high": [1.9, 5.0, np.nan, 2.7],
            "n_mice_control": [2, 2, 1, 2],
            "n_mice_irradiated": [2, 2, 2, 2],
        }
    )
    align = pd.DataFrame(
        {
            "leiden": ["d", "b", "c", "a"],
            "alignment_cosine": [-0.8, 0.1, np.nan, 0.9],
            "usable": [True, True, False, True],
            "rep": ["X_pca"] * 4,
        }
    )
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
    assert (
        not table.ci_low.notna().any()
    )  # raw CI must not appear for normalized effect
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
    assert result.metadata["interval_columns"] == [
        "magnitude_ci95_low",
        "magnitude_ci95_high",
    ]
    plt.close(result.fig)


def test_concordance_rejects_duplicate_ids_invalid_interval_and_cosine(geometry_tables):
    shift, align = geometry_tables
    with pytest.raises(ValueError, match="duplicate"):
        sg.pl.geometry_velocity_concordance(
            pd.concat([shift, shift.iloc[:1]]), align, show=False
        )
    bad = align.copy()
    bad.loc[0, "alignment_cosine"] = 1.5
    with pytest.raises(ValueError, match="between -1 and 1"):
        sg.pl.geometry_velocity_concordance(shift, bad, show=False)
    bad_shift = shift.copy()
    bad_shift.loc[0, "magnitude_ci95_low"] = 2.0
    with pytest.raises(ValueError, match="ordered"):
        sg.pl.geometry_velocity_concordance(
            bad_shift, align, effect_metric="delta_norm", show=False
        )


def test_concordance_retains_effect_and_interval_without_alignment(geometry_tables):
    shift, align = geometry_tables
    align.loc[align.leiden.eq("a"), ["alignment_cosine", "usable"]] = [np.nan, False]
    result = sg.pl.geometry_velocity_concordance(
        shift,
        align,
        effect_metric="delta_norm",
        return_result=True,
        show=False,
    )
    table = result.data["states"].set_index("_state_id")
    assert table.loc["a", "effect_eligible"]
    assert not table.loc["a", "alignment_eligible"]
    assert table.loc["a", "ci_low"] == pytest.approx(1.2)
    assert len(result.artists["effects"]) == 3 and len(result.artists["points"]) == 2
    labels = [label.get_text() for label in result.axes["effect"].get_yticklabels()]
    assert "a · alignment unavailable" in labels and "c" in labels
    assert len(result.artists["unavailable"]) == 1
    low, high = result.axes["effect"].get_ylim()
    assert low == pytest.approx(3.5) and high == pytest.approx(-0.5)
    plt.close(result.fig)


def test_concordance_metric_status_and_explicit_support_are_independent(
    geometry_tables,
):
    shift, align = geometry_tables
    shift["effect_status"] = "assessed"
    shift["n_cells0"] = [100, 100, 1, 100]
    shift["n_cells1"] = 100
    align["n_cells0"], align["n_cells1"] = 100, 100
    align["n_samples0"], align["n_samples1"] = 2, 2
    shift.loc[shift.state.eq("a"), "coverage_status"] = "insufficient_coverage"
    shift.loc[shift.state.eq("c"), "normalized_delta_norm"] = 5.0
    result = sg.pl.geometry_velocity_concordance(
        shift,
        align,
        min_cells=10,
        min_samples=2,
        return_result=True,
        show=False,
    )
    table = result.data["states"].set_index("_state_id")
    assert table.loc[
        "a", "eligible"
    ]  # combined coverage need not suppress a valid effect
    assert table.loc["c", "effect"] == 5.0  # preserve the recorded number
    assert not table.loc["c", "effect_eligible"]
    assert table.loc["c", "effect_reason"] == "insufficient_cells"
    assert result.metadata["support_thresholds"]["min_cells"]["effective"] == 10
    plt.close(result.fig)


def test_concordance_source_floor_cannot_be_relaxed_by_display(geometry_tables):
    shift, align = geometry_tables
    shift["n_cells0"], shift["n_cells1"] = [2, 40, 1, 40], 40
    shift.attrs["params"] = {"min_cells": 10}
    result = sg.pl.geometry_velocity_concordance(
        shift,
        align,
        min_cells=1,
        return_result=True,
        show=False,
    )
    assert not result.data["states"].set_index("_state_id").loc["a", "effect_eligible"]
    threshold = result.metadata["support_thresholds"]["min_cells"]
    assert (
        threshold["recorded"] == 10
        and threshold["requested"] == 1
        and threshold["effective"] == 10
    )
    plt.close(result.fig)


def test_concordance_unknown_counts_do_not_satisfy_explicit_floor(geometry_tables):
    shift, align = geometry_tables
    result = sg.pl.geometry_velocity_concordance(
        shift,
        align,
        min_cells=10,
        return_result=True,
        show=False,
    )
    table = result.data["states"].set_index("_state_id")
    assert table.loc["a", "effect_reason"] == "cells_counts_unverified"
    assert not table.effect_eligible.any()
    assert len(result.axes["effect"].get_yticklabels()) == 4
    plt.close(result.fig)


@pytest.mark.parametrize("threshold_source", ["explicit", "recorded"])
def test_concordance_alignment_source_support_is_checked_independently(
    geometry_tables, threshold_source
):
    shift, align = geometry_tables
    shift["n_cells0"], shift["n_cells1"] = 100, 100
    align["n_cells0"], align["n_cells1"] = 100, 100
    align.loc[align.leiden.eq("a"), "n_cells0"] = 1
    if threshold_source == "recorded":
        align.attrs["params"] = {"min_cells": 10}
    result = sg.pl.geometry_velocity_concordance(
        shift,
        align,
        min_cells=10 if threshold_source == "explicit" else None,
        return_result=True,
        show=False,
    )
    row = result.data["states"].set_index("_state_id").loc["a"]
    assert row.effect_eligible and not row.alignment_eligible
    assert row.alignment == pytest.approx(0.9)
    assert row.alignment_reason == "alignment_insufficient_cells"
    assert (
        result.metadata["alignment_support_thresholds"]["min_cells"]["effective"] == 10
    )
    plt.close(result.fig)


def test_concordance_never_borrows_shift_counts_to_certify_alignment(geometry_tables):
    shift, align = geometry_tables
    shift["n_cells0"], shift["n_cells1"] = 100, 100
    result = sg.pl.geometry_velocity_concordance(
        shift,
        align,
        min_cells=10,
        return_result=True,
        show=False,
    )
    row = result.data["states"].set_index("_state_id").loc["a"]
    assert row.effect_eligible and not row.alignment_eligible
    assert row.alignment_reason == "alignment_cells_counts_unverified"
    assert row.alignment_cells_count_status == "unverified"
    plt.close(result.fig)


def test_concordance_descriptive_effects_keep_source_ci_but_do_not_draw_it(
    geometry_tables,
):
    shift, align = geometry_tables
    shift["descriptive_only"] = [True, False, False, False]
    shift["inference_level"] = [
        "descriptive",
        "biological_sample",
        "not_computed",
        "sample_descriptive",
    ]
    result = sg.pl.geometry_velocity_concordance(
        shift,
        align,
        effect_metric="delta_norm",
        return_result=True,
        show=False,
    )
    table = result.data["states"].set_index("_state_id")
    assert table.loc["a", "effect_eligible"] and table.loc["d", "effect_eligible"]
    assert table.loc["a", "magnitude_ci95_low"] == pytest.approx(1.2)
    assert np.isnan(table.loc["a", "ci_low"]) and np.isnan(table.loc["d", "ci_high"])
    assert table.loc["a", "interval_reason"] == "descriptive_only"
    assert result.metadata["n_intervals_displayed"] == 1
    assert any(
        "d · descriptive" == label.get_text()
        for label in result.axes["effect"].get_yticklabels()
    )
    plt.close(result.fig)


def test_concordance_display_labels_preserve_distinct_ids_when_text_repeats(
    geometry_tables,
):
    shift, align = geometry_tables
    shift["name"] = ["Shared name", "Shared name", "Rare state", "Other state"]
    result = sg.pl.geometry_velocity_concordance(
        shift,
        align,
        label_key="name",
        max_labels=1,
        highlight_states=["a", "b", "c"],
        label_offsets={"a": (20, 15)},
        return_result=True,
        show=False,
    )
    assert set(result.artists["labels"]) == {"a", "b"}
    assert all(
        artist.get_text() == "Shared name"
        for artist in result.artists["labels"].values()
    )
    assert result.metadata["highlight_label_overflow"] == 1
    assert result.metadata["highlighted_unavailable"] == ["c"]
    placement = result.data["label_placement"].set_index("state")
    assert placement.loc["a", "offset_x_points"] == 20
    assert placement.loc["a", "placement"] == "manual"
    assert "Rare state" in [
        label.get_text() for label in result.axes["effect"].get_yticklabels()
    ]
    plt.close(result.fig)


def test_concordance_label_budget_is_stable_and_highlights_have_priority(
    geometry_tables,
):
    shift, align = geometry_tables
    results = []
    for frame in (shift, shift.iloc[::-1]):
        result = sg.pl.geometry_velocity_concordance(
            frame,
            align,
            max_labels=2,
            highlight_states=["d"],
            return_result=True,
            show=False,
        )
        assert result.metadata["labelled_states"] == ["d", "b"]
        assert result.metadata["omitted_labels"] == ["a"]
        results.append(result)
    pd.testing.assert_frame_equal(
        results[0].data["label_placement"], results[1].data["label_placement"]
    )
    for result in results:
        plt.close(result.fig)


def test_concordance_rejects_reused_axes_before_mutation(geometry_tables):
    shift, align = geometry_tables
    fig, ax = plt.subplots()
    before = len(ax.get_children())
    with pytest.raises(ValueError, match="distinct Axes"):
        sg.pl.geometry_velocity_concordance(
            shift,
            align,
            axes={"concordance": ax, "effect": ax},
            show=False,
        )
    assert len(ax.get_children()) == before
    plt.close(fig)


def test_concordance_rejects_cross_source_provenance_before_axes_mutation(
    geometry_tables,
):
    shift, align = geometry_tables
    shift.attrs["provenance"] = {"effect_representation": "X_other"}
    fig, array = plt.subplots(1, 2)
    before = [len(ax.get_children()) for ax in array]
    with pytest.raises(ValueError, match="analysis_rep"):
        sg.pl.geometry_velocity_concordance(
            shift,
            align,
            axes=dict(zip(("concordance", "effect"), array)),
            show=False,
        )
    assert [len(ax.get_children()) for ax in array] == before
    plt.close(fig)


def test_concordance_alignment_only_state_remains_visible(geometry_tables):
    shift, align = geometry_tables
    align = pd.concat(
        [
            align,
            pd.DataFrame(
                {
                    "leiden": ["only_alignment"],
                    "alignment_cosine": [0.7],
                    "usable": [True],
                    "rep": ["X_pca"],
                }
            ),
        ],
        ignore_index=True,
    )
    result = sg.pl.geometry_velocity_concordance(
        shift, align, return_result=True, show=False
    )
    table = result.data["states"].set_index("_state_id")
    assert table.loc["only_alignment", "alignment_eligible"]
    assert not table.loc["only_alignment", "effect_eligible"]
    assert table.loc["only_alignment", "reason"] == "effect_missing"
    assert "only_alignment" in [
        label.get_text() for label in result.axes["effect"].get_yticklabels()
    ]
    plt.close(result.fig)


@pytest.mark.parametrize(
    "zero_kind",
    [
        "shift",
        "velocity",
        "declared_shift",
        "shift_raw_norm",
        "alignment_raw_norm",
        "alignment_normalized_norm",
    ],
)
def test_concordance_zero_vectors_never_create_direction(geometry_tables, zero_kind):
    shift, align = geometry_tables
    if zero_kind == "shift":
        shift.loc[shift.state.eq("a"), "normalized_delta_norm"] = 0.0
    elif zero_kind == "declared_shift":
        shift["effect_status"] = "assessed"
        shift.loc[shift.state.eq("a"), "effect_status"] = "zero_shift"
    elif zero_kind == "shift_raw_norm":
        shift.loc[shift.state.eq("a"), "delta_norm"] = 0.0
    elif zero_kind in {"alignment_raw_norm", "alignment_normalized_norm"}:
        column = (
            "delta_norm"
            if zero_kind == "alignment_raw_norm"
            else "normalized_delta_norm"
        )
        align[column] = 1.0
        align.loc[align.leiden.eq("a"), column] = 0.0
    else:
        align["vel_norm"] = 1.0
        align.loc[align.leiden.eq("a"), "vel_norm"] = 0.0
    result = sg.pl.geometry_velocity_concordance(
        shift, align, return_result=True, show=False
    )
    row = result.data["states"].set_index("_state_id").loc["a"]
    assert row.effect_eligible and not row.alignment_eligible
    assert row.alignment_reason == (
        "zero_velocity" if zero_kind == "velocity" else "zero_shift"
    )
    assert row.alignment == pytest.approx(0.9)  # raw supplied value remains auditable
    plt.close(result.fig)


def test_concordance_stored_estimates_ignore_display_columns(geometry_tables):
    shift, align = geometry_tables
    before = sg.pl.geometry_velocity_concordance(
        shift, align, return_result=True, show=False
    )
    shift["umap_x"], shift["umap_y"] = [9, -100, np.nan, 1e8], [2, 3, 4, 5]
    shift.attrs["display_basis"] = "unrelated_embedding"
    after = sg.pl.geometry_velocity_concordance(
        shift, align, return_result=True, show=False
    )
    columns = ["_state_id", "effect", "alignment", "eligible", "ci_low", "ci_high"]
    pd.testing.assert_frame_equal(
        before.data["states"][columns], after.data["states"][columns]
    )
    for result in (before, after):
        plt.close(result.fig)


def test_concordance_known_single_sample_withholds_ci_without_descriptive_flag(
    geometry_tables,
):
    shift, align = geometry_tables
    shift.loc[shift.state.eq("a"), "n_mice_control"] = 1
    result = sg.pl.geometry_velocity_concordance(
        shift,
        align,
        effect_metric="delta_norm",
        return_result=True,
        show=False,
    )
    row = result.data["states"].set_index("_state_id").loc["a"]
    assert row.effect_eligible and row.descriptive_only_known
    assert row.magnitude_ci95_low == pytest.approx(1.2) and np.isnan(row.ci_low)
    assert row.interval_reason == "descriptive_only"
    plt.close(result.fig)


@pytest.mark.parametrize(
    "location,unit_key",
    [
        ("column", "bootstrap_unit"),
        ("column", "resolved_bootstrap_unit"),
        ("attrs", "bootstrap_unit"),
        ("attrs", "resolved_bootstrap_unit"),
    ],
)
def test_concordance_cell_bootstrap_withholds_ci_with_two_samples(
    geometry_tables, location, unit_key
):
    shift, align = geometry_tables
    if location == "column":
        shift[unit_key] = pd.Series("sample", index=shift.index, dtype="string")
        shift.loc[shift.state.eq("a"), unit_key] = "cell"
        shift.loc[shift.state.eq("b"), unit_key] = pd.NA
    else:
        shift.attrs["provenance"] = {"params": {unit_key: "cell"}}
    before = shift.copy(deep=True)
    result = sg.pl.geometry_velocity_concordance(
        shift,
        align,
        effect_metric="delta_norm",
        return_result=True,
        show=False,
    )
    table = result.data["states"].set_index("_state_id")
    row = table.loc["a"]
    assert row.n_mice_control == 2 and row.n_mice_irradiated == 2
    assert (
        row.effect_eligible and row.cell_bootstrap_known and row.descriptive_only_known
    )
    assert row.magnitude_ci95_low == pytest.approx(
        1.2
    ) and row.magnitude_ci95_high == pytest.approx(1.9)
    assert not row.interval_displayed and np.isnan(row.ci_low) and np.isnan(row.ci_high)
    assert row.interval_reason == "descriptive_only"
    if location == "column":
        assert table.loc["b", "interval_displayed"]
        assert not table.loc["b", "cell_bootstrap_known"]
    pd.testing.assert_frame_equal(shift, before)
    assert shift.attrs == before.attrs
    plt.close(result.fig)


@pytest.fixture
def mouse_like_adata():
    rng = np.random.RandomState(18)
    n = 240
    labels = np.array(["reference"] * n + ["treated"] * n, dtype=object)
    xy = np.vstack(
        [
            rng.normal(loc=(0, 0), scale=(0.45, 0.65), size=(n, 2)),
            rng.normal(loc=(0.14, -0.09), scale=(0.45, 0.65), size=(n, 2)),
        ]
    )
    vel = np.column_stack([np.ones(n * 2), np.zeros(n * 2)]) * 0.12
    vel[n:, 0] = -0.12
    obs = pd.DataFrame(
        {
            "condition": labels,
            "state": pd.Categorical(
                ["A"] * (n // 2)
                + ["B"] * (n // 2)
                + ["A"] * (n // 2)
                + ["B"] * (n // 2)
            ),
        },
        index=[f"c{i}" for i in range(2 * n)],
    )
    adata = AnnData(np.zeros((2 * n, 2)), obs=obs)
    adata.obsm["X_umap"] = xy
    adata.obsm["velocity_umap"] = vel
    adata.uns["paga"] = {
        "groups": "state",
        "connectivities": sparse.csr_matrix([[0, 0.8], [0.8, 0]]),
    }
    yield adata
    plt.close("all")


def test_velocity_conditions_share_grid_and_do_not_pool_opposing_vectors(
    mouse_like_adata,
):
    adata = mouse_like_adata
    before = adata.obsm["velocity_umap"].copy()
    out = sg.pl.velocity_condition_compare(
        adata,
        condition_key="condition",
        group0="reference",
        group1="treated",
        grid_size=13,
        radius=0.65,
        n_neighbors=25,
        min_neighbors=5,
        min_coherence=0.1,
        return_result=True,
        show=False,
    )
    table = out.data["paired_grid"]
    both = table.loc[table.support_status == "both_supported"]
    assert len(both) > 0
    assert (both.direction_cosine < -0.95).all()
    assert np.array_equal(
        out.data["reference_grid"][["x", "y"]].to_numpy(),
        out.data["comparison_grid"][["x", "y"]].to_numpy(),
    )
    assert np.allclose(
        out.data["reference_grid"].loc[out.data["reference_grid"].supported, "vx"], 0.12
    )
    assert np.allclose(
        out.data["comparison_grid"].loc[out.data["comparison_grid"].supported, "vx"],
        -0.12,
    )
    assert out.metadata["shared_scale"] > 0
    np.testing.assert_array_equal(adata.obsm["velocity_umap"], before)
    plt.close(out.fig)


def test_velocity_comparison_streamlines_and_absent_velocity(mouse_like_adata):
    adata = mouse_like_adata
    out = sg.pl.velocity_condition_compare(
        adata,
        condition_key="condition",
        group0="reference",
        group1="treated",
        grid_size=12,
        radius=0.7,
        n_neighbors=30,
        min_neighbors=4,
        mode="stream",
        stream_density=1.0,
        return_result=True,
        show=False,
    )
    assert len(out.data["paired_grid"]) == 144
    assert out.metadata["n_both_supported"] > 0
    plt.close(out.fig)
    del adata.obsm["velocity_umap"]
    with pytest.raises(KeyError, match="velocity_umap"):
        sg.pl.velocity_condition_compare(
            adata,
            condition_key="condition",
            group0="reference",
            group1="treated",
            show=False,
        )


def test_paga_focus_only_changes_display_and_keeps_original_graph(mouse_like_adata):
    adata = mouse_like_adata
    original = adata.uns["paga"]["connectivities"].copy()
    result = sg.pl.paga_shift_map(
        adata,
        node_key="state",
        condition_key="condition",
        group0="reference",
        group1="treated",
        min_cells=1,
        focus_states=["A"],
        edge_focus="within",
        label_top_n=0,
        return_result=True,
        show=False,
    )
    table = result.data["edge_table"]
    assert table.displayed.sum() == 0
    assert table.reason.str.contains("outside_focus").any()
    assert result.metadata["focus_states"] == ["A"]
    assert result.metadata["edge_focus"] == "within"
    assert np.array_equal(
        adata.uns["paga"]["connectivities"].toarray(), original.toarray()
    )
    plt.close(result.fig)
    with pytest.raises(ValueError, match="focus_states"):
        sg.pl.paga_shift_map(
            adata,
            node_key="state",
            condition_key="condition",
            group0="reference",
            group1="treated",
            focus_states=["missing"],
            edge_focus="within",
            show=False,
        )
