"""Compass evidence stays separate from observed coverage and drawing choices."""

from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

import scgeo as sg


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def data():
    rows, xy, velocity = [], [], []
    for sample, condition, enriched in (
        ("C1", "control", 0),
        ("C2", "control", 1),
        ("T1", "treated", 5),
        ("T2", "treated", 10),
    ):
        for state, count in (("Shared", 8), ("Enriched", enriched)):
            for i in range(count):
                rows.append(
                    {
                        "state": state,
                        "condition": condition,
                        "sample": sample,
                        "ood": 0.1 if state == "Shared" else 0.9,
                    }
                )
                xy.append(
                    [
                        float(state == "Enriched") * 5
                        + float(condition == "treated")
                        + i / 50,
                        float(state == "Enriched") * 2 + i / 100,
                    ]
                )
                velocity.append([1.0, 0.0])
    for i in range(4):
        rows.append(
            {"state": "Shared", "condition": "other", "sample": "O1", "ood": 100.0}
        )
        xy.append([i / 20, 0.0])
        velocity.append([-100.0, 0.0])
    obs = pd.DataFrame(rows, index=[f"cell_{i}" for i in range(len(rows))])
    obs["state"] = pd.Categorical(
        obs.state, categories=["Shared", "Enriched", "Absent"]
    )
    a = AnnData(np.zeros((len(obs), 1)), obs=obs)
    a.obsm["X_umap"] = np.array(xy)
    a.obsm["velocity_umap"] = np.array(velocity)
    a.uns["paga"] = {
        "groups": "state",
        "node_order": ["Shared", "Enriched", "Absent"],
        "connectivities": np.array([[0, 0.8, 0.2], [0.8, 0, 0], [0.2, 0, 0.0]]),
    }
    return a


def plot(a, **kwargs):
    options = {
        "node_key": "state",
        "condition_key": "condition",
        "group0": "control",
        "group1": "treated",
        "min_cells": 2,
        "node_position": "pooled",
        "legend": False,
        "show": False,
        "return_result": True,
    }
    options.update(kwargs)
    return sg.pl.recovery_compass(a, **options)


def reports():
    params = {
        "rep": "X_pca",
        "node_key": "state",
        "condition_key": "condition",
        "group0": "control",
        "group1": "treated",
    }
    shifts = pd.DataFrame(
        {
            "state": ["Shared", "Enriched", "Absent"],
            "delta_norm": [2.0, 3.0, 1.0],
            "normalized_delta_norm": [1.0, 1.5, 0.5],
            "magnitude_ci95_low": [1.5, 2.0, 0.5],
            "magnitude_ci95_high": [2.5, 4.0, 1.5],
            "effect_status": ["assessed"] * 3,
            "coverage_status": ["insufficient_coverage"] * 3,
            "n_cells0": [16, 1, 10],
            "n_cells1": [16, 15, 10],
            "n_samples0": [2, 1, 2],
            "n_samples1": [2, 2, 2],
            "descriptive_only": [False, True, False],
        }
    )
    alignments = pd.DataFrame(
        {
            "node": ["Absent", "Shared", "Enriched"],
            "alignment_cosine": [-0.4, 0.7, 0.1],
            "usable": [True, True, False],
        }
    )
    shifts.attrs["params"] = copy.deepcopy(params)
    alignments.attrs["params"] = copy.deepcopy(params)
    return shifts, alignments


def test_contrast_scope_is_invariant_to_unrelated_condition_velocities(data):
    result = plot(data)
    all_conditions = plot(data, velocity_scope="all")
    nodes = result.data["nodes"].set_index("state")
    legacy = all_conditions.data["nodes"].set_index("state")
    assert nodes.loc["Shared", "alignment_cosine"] > 0.99
    assert legacy.loc["Shared", "alignment_cosine"] < -0.99
    assert nodes.loc["Shared", "n_vel"] == 32
    assert legacy.loc["Shared", "n_vel"] == 36
    for scope, n in (("group0", 16), ("group1", 16)):
        assert (
            plot(data, velocity_scope=scope)
            .data["nodes"]
            .set_index("state")
            .loc["Shared", "n_vel"]
            == n
        )


def test_incomplete_vectors_never_form_a_synthetic_mean(data):
    selected = data.obs.state.eq("Shared") & data.obs.condition.ne("other")
    data.obsm["velocity_umap"][selected] = np.tile(
        [[1.0, np.nan], [np.nan, 1.0]], (16, 1)
    )
    result = plot(data)
    row = result.data["nodes"].set_index("state").loc["Shared"]
    assert row.n_vel == 0 and row.n_vel_missing == 32
    assert np.isnan(row.vx) and np.isnan(row.vy) and np.isnan(row.alignment_cosine)
    assert not row.usable_velocity
    assert not result.data["glyphs"].set_index("node").loc["Shared", "fill_available"]


def test_wrong_projection_and_nonfinite_selected_coordinates_fail_before_drawing(data):
    _, ax = plt.subplots()
    before = list(ax.get_children())
    data.obsm["velocity_tsne"] = data.obsm["velocity_umap"].copy()
    with pytest.raises(ValueError, match="match the display"):
        plot(data, velocity_basis="tsne", ax=ax)
    assert list(ax.get_children()) == before
    data.obsm["X_umap"][0, 1] = np.nan
    with pytest.raises(ValueError, match="jointly finite"):
        plot(data, ax=ax)
    assert list(ax.get_children()) == before


def test_explicit_missing_velocity_is_an_error_but_automatic_missing_is_recorded(data):
    del data.obsm["velocity_umap"]
    result = plot(data)
    assert result.data["nodes"].n_vel.eq(0).all()
    assert any("alignment remains missing" in w for w in result.warnings)
    with pytest.raises(KeyError, match="Explicit"):
        plot(data, velocity_basis="umap")


def test_zero_shift_and_zero_display_scale_produce_no_arrow(data):
    data.obsm["X_umap"][data.obs.state.eq("Shared")] = [0.0, 0.0]
    result = plot(data)
    row = result.data["glyphs"].set_index("node").loc["Shared"]
    assert not row.arrow_displayed and row.arrow_reason == "zero_shift"
    assert "Shared" not in result.artists["arrows"]
    hidden = plot(data, arrow_scale=0)
    assert not hidden.artists["arrows"]
    assert hidden.data["glyphs"].arrow_reason.eq("zero_scale").all()


def test_ood_missing_scores_and_measured_zero_have_distinct_encodings(data):
    data.obs.loc[data.obs.state.eq("Enriched"), "ood"] = np.nan
    result = plot(data, ood_key="ood", ood_threshold=0.5)
    nodes, glyphs = (
        result.data["nodes"].set_index("state"),
        result.data["glyphs"].set_index("node"),
    )
    assert nodes.loc["Shared", "ood_frac"] == 0
    assert glyphs.loc["Shared", "ring_available"]
    assert np.isnan(nodes.loc["Enriched", "ood_frac"])
    assert nodes.loc["Enriched", "ood_n_scored"] == 0
    assert nodes.loc["Enriched", "ood_n_missing"] == 16
    assert not glyphs.loc["Enriched", "ring_available"]
    assert (
        result.data["cells"]
        .loc[result.data["cells"].state.eq("Enriched"), "ood_flagged"]
        .isna()
        .all()
    )
    expected = np.quantile(
        data.obs.loc[data.obs.condition.ne("other"), "ood"].dropna(), 0.95
    )
    automatic = plot(data, ood_key="ood")
    assert automatic.metadata["ood_threshold"] == expected
    assert (
        plot(data, ood_key="ood", ood_scope="all").metadata["ood_threshold"] > expected
    )


def test_all_ood_scores_missing_remain_unavailable(data):
    data.obs["ood"] = np.nan
    result = plot(data, ood_key="ood")
    assert result.data["nodes"].ood_frac.isna().all()
    assert not result.data["glyphs"].ring_available.any()


@pytest.mark.parametrize("remove_control", [False, True])
def test_pooled_sparse_nodes_keep_topology_without_inventing_reference(
    data, remove_control
):
    if remove_control:
        data = data[
            ~(data.obs.state.eq("Enriched") & data.obs.condition.eq("control"))
        ].copy()
    result = plot(data)
    row = result.data["nodes"].set_index("state").loc["Enriched"]
    glyph = result.data["glyphs"].set_index("node").loc["Enriched"]
    assert row.displayed and glyph.displayed and "Enriched" in result.artists["nodes"]
    assert not glyph.arrow_displayed and not glyph.fill_available
    assert (
        result.data["edges"]
        .set_index(["source", "target"])
        .loc[("Shared", "Enriched"), "displayed"]
    )
    if remove_control:
        assert np.isnan(row.x0) and np.isnan(glyph.arrow_source_x)
    reference = plot(data, node_position="reference")
    for column in ("x0", "y0", "x1", "y1", "dx", "dy"):
        np.testing.assert_allclose(
            result.data["nodes"][column],
            reference.data["nodes"][column],
            equal_nan=True,
        )


def test_summary_uses_complete_sample_roster_with_zero_state_frequencies(data):
    result = plot(data, sample_key="sample", layout="summary")
    table = result.data["sample_abundance"].set_index(["state", "condition", "sample"])
    assert table.loc[("Enriched", "control", "C1"), "count"] == 0
    assert table.loc[("Enriched", "control", "C1"), "denominator"] == 8
    assert table.loc[("Enriched", "control", "C1"), "state_fraction"] == 0
    assert table.loc[("Enriched", "control", "C2"), "state_fraction"] == pytest.approx(
        1 / 9
    )
    assert "O1" not in result.data["sample_abundance"]["sample"].unique()
    coverage = result.data["coverage"].set_index(["state", "condition"])
    assert coverage.loc[("Enriched", "control"), "n_samples"] == 1
    assert coverage.loc[("Enriched", "control"), "n_samples_total"] == 2
    assert {"map", "effect", "alignment", "abundance", "coverage"} <= set(result.axes)
    pooled = plot(data, layout="summary")
    assert pooled.data["sample_abundance"].empty
    assert pooled.data["coverage"].n_samples.isna().all()


def test_stored_effect_survives_missing_alignment_and_current_sparse_context(data):
    shifts, _ = reports()
    result = plot(
        data,
        mode="stored",
        shift_df=shifts,
        analysis_rep="X_pca",
        effect_metric="delta_norm",
        layout="summary",
    )
    evidence = result.data["evidence"].set_index("state")
    assert evidence.loc["Shared", "effect_available"]
    assert evidence.loc["Shared", "interval_available"]
    assert not evidence.alignment_available.any()
    assert evidence.loc[
        "Absent", "effect_available"
    ]  # source counts, not current context
    assert not result.data["nodes"].set_index("state").loc["Absent", "displayed"]
    assert not evidence.loc["Enriched", "effect_available"]
    assert (
        evidence.loc["Enriched", "effect_support_status"] == "source_insufficient_cells"
    )
    assert evidence.loc["Enriched", "effect"] == 3.0


def test_stored_values_and_eligibility_are_independent_of_display_coordinates(data):
    shifts, alignments = reports()
    before, before_alignment = shifts.copy(deep=True), alignments.copy(deep=True)
    first = plot(
        data,
        mode="stored",
        shift_df=shifts,
        alignment_df=alignments,
        analysis_rep="X_pca",
    )
    data.obsm["X_umap"] = data.obsm["X_umap"] @ np.array([[2.0, 1.0], [-1.0, 3.0]]) + [
        20.0,
        -5.0,
    ]
    data.obsm["velocity_umap"] = data.obsm["velocity_umap"] @ np.array(
        [[2.0, 1.0], [-1.0, 3.0]]
    )
    second = plot(
        data,
        mode="stored",
        shift_df=shifts,
        alignment_df=alignments,
        analysis_rep="X_pca",
    )
    pd.testing.assert_frame_equal(first.data["evidence"], second.data["evidence"])
    np.testing.assert_allclose(
        first.data["glyphs"].fill_value,
        second.data["glyphs"].fill_value,
        equal_nan=True,
    )
    assert first.metadata["effect_metric"] == "normalized_delta_norm"
    assert not first.data[
        "evidence"
    ].interval_available.any()  # raw intervals not borrowed
    assert first.metadata["stored_provenance"]["status"] == "recorded_consistent"
    pd.testing.assert_frame_equal(shifts, before)
    pd.testing.assert_frame_equal(alignments, before_alignment)


def test_source_low_samples_stay_descriptive_with_no_interval(data):
    shifts, _ = reports()
    shifts.loc[shifts.state.eq("Enriched"), "n_cells0"] = 10
    result = plot(
        data,
        mode="stored",
        shift_df=shifts,
        analysis_rep="X_pca",
        effect_metric="delta_norm",
    )
    row = result.data["evidence"].set_index("state").loc["Enriched"]
    assert row.effect_available and row.effect_descriptive and not row.effect_supported
    assert not row.interval_available and row.ci_low == 2.0
    assert row.effect_support_status == "source_insufficient_samples"


@pytest.mark.parametrize(
    "change", ["contradiction", "ambiguous_state", "negative_effect", "half_interval"]
)
def test_invalid_stored_evidence_fails_before_mutating_axes(data, change):
    shifts, alignment = reports()
    if change == "contradiction":
        alignment.attrs["params"]["group0"] = "other"
    elif change == "ambiguous_state":
        shifts["node"] = ["X", "Y", "Z"]
    elif change == "negative_effect":
        shifts.loc[0, "delta_norm"] = -1
    else:
        shifts = shifts.drop(columns="magnitude_ci95_high")
    _, ax = plt.subplots()
    before = list(ax.get_children())
    with pytest.raises((ValueError, KeyError)):
        plot(
            data,
            mode="stored",
            shift_df=shifts,
            alignment_df=alignment,
            analysis_rep="X_pca",
            effect_metric="delta_norm",
            ax=ax,
        )
    assert list(ax.get_children()) == before


def test_shared_scales_sizing_and_point_arrows_are_honored(data):
    fill, arrow = Normalize(-1, 1), Normalize(0, 3)
    result = plot(
        data,
        fill_norm=fill,
        arrow_norm=arrow,
        show_colorbar=True,
        node_size_reference=100,
    )
    assert result.artists["fill_colorbar"].mappable.norm is fill
    assert result.artists["arrow_colorbar"].mappable.norm is arrow
    assert all(
        isinstance(a, FancyArrowPatch) for a in result.artists["arrows"].values()
    )
    assert result.metadata["arrow_width_units"] == "points"
    row = result.data["glyphs"].set_index("node").loc["Shared"]
    assert row["size"] == pytest.approx(380 * (0.45 + 0.55 * np.sqrt(16 / 100)))
    delta = plot(data, fill_color_mode="delta_frac", fill_vmin=-0.5, fill_vmax=0.5)
    assert delta.metadata["fill_norm"]["vmin"] == -0.5
    assert delta.metadata["fill_norm"]["vmax"] == 0.5


def test_display_sampling_preserves_all_summaries(data):
    full = plot(data, sample_key="sample")
    sampled = plot(data, sample_key="sample", max_points=5)
    assert sampled.data["cells"].background_displayed.sum() == 5
    assert len(sampled.data["cells"]) == data.n_obs
    for table in ("nodes", "coverage", "sample_abundance", "evidence", "edges"):
        pd.testing.assert_frame_equal(full.data[table], sampled.data[table])
    assert sampled.axes["map"].get_xlim() == full.axes["map"].get_xlim()


def test_summary_caller_axes_and_repeated_label_names_preserve_identity(data, tmp_path):
    fig, array = plt.subplots(1, 5, figsize=(16, 4))
    axes = dict(zip(("map", "effect", "alignment", "abundance", "coverage"), array))
    fig.suptitle("Caller title")
    before = [a.get_position().bounds for a in array]
    result = plot(
        data,
        layout="summary",
        axes=axes,
        sample_key="sample",
        label_names={"Shared": "Same", "Enriched": "Same"},
    )
    np.testing.assert_allclose([a.get_position().bounds for a in array[1:]], before[1:])
    assert fig._suptitle.get_text() == "Caller title"
    assert set(result.artists["labels"]) == {"Shared", "Enriched"}
    assert all(text.get_text() == "Same" for text in result.artists["labels"].values())
    output = result.save(tmp_path / "compass", formats="svg")
    assert {
        "cells",
        "nodes",
        "edges",
        "graph_nodes",
        "glyphs",
        "coverage",
        "sample_abundance",
        "evidence",
    } <= set(output["tables"])


@pytest.mark.parametrize("source", ["shift", "alignment"])
def test_known_zero_source_shift_has_no_stored_alignment(data, source):
    shifts, alignment = reports()
    frame = shifts if source == "shift" else alignment
    key = "state" if source == "shift" else "node"
    frame.loc[frame[key].eq("Shared"), "delta_norm"] = 0.0
    result = plot(
        data,
        mode="stored",
        shift_df=shifts,
        alignment_df=alignment,
        analysis_rep="X_pca",
    )
    evidence = result.data["evidence"].set_index("state")
    assert evidence.loc["Shared", "alignment"] == 0.7
    assert not evidence.loc["Shared", "alignment_available"]
    assert evidence.loc["Shared", "alignment_reason"] == "undefined_zero_shift"
    assert not result.data["glyphs"].set_index("node").loc["Shared", "fill_available"]


def test_alignment_support_uses_its_source_counts_not_current_map_counts(data):
    shifts, alignment = reports()
    alignment["n_cells0"] = [10, 1, 20]
    alignment["n_cells1"] = [10, 20, 20]
    alignment["usable"] = True
    result = plot(
        data,
        mode="stored",
        shift_df=shifts,
        alignment_df=alignment,
        analysis_rep="X_pca",
    )
    table = result.data["evidence"].set_index("state")
    assert table.loc["Shared", "effect_available"]
    assert not table.loc["Shared", "alignment_available"]
    assert table.loc["Shared", "alignment_reason"] == "source_insufficient_cells"
    assert table.loc["Enriched", "alignment_available"]
    assert not table.loc["Enriched", "effect_available"]


def test_unannotated_sample_denominator_remains_missing_and_sample_identity_visible(
    data,
):
    data.obs.loc[data.obs["sample"].eq("C1"), "state"] = np.nan
    result = plot(data, layout="summary", sample_key="sample")
    missing = result.data["sample_abundance"].loc[
        result.data["sample_abundance"]["sample"].eq("C1")
    ]
    assert missing.denominator.eq(0).all()
    assert missing.state_fraction.isna().all() and not missing.included.any()
    legend = result.artists["summary"]["sample_legend"]
    assert {t.get_text() for t in legend.get_texts()} == {"C1", "C2", "T1", "T2"}
    assert len({h.get_marker() for h in legend.legend_handles}) == 4


@pytest.mark.parametrize(
    "inference",
    [
        "sample_descriptive",
        "cell_descriptive",
        "cell_bootstrap",
        "status_insufficient_samples",
    ],
)
def test_known_descriptive_design_without_boolean_flag_withholds_intervals(
    data, inference
):
    shifts, _ = reports()
    shifts = shifts.drop(columns="descriptive_only")
    if inference == "cell_bootstrap":
        shifts.attrs["params"]["bootstrap_unit"] = "cell"
    elif inference == "status_insufficient_samples":
        shifts.loc[shifts.state.eq("Shared"), "effect_status"] = "insufficient_samples"
    else:
        shifts["inference_level"] = inference
    result = plot(
        data,
        mode="stored",
        shift_df=shifts,
        analysis_rep="X_pca",
        effect_metric="delta_norm",
    )
    row = result.data["evidence"].set_index("state").loc["Shared"]
    assert row.effect_available and row.effect_descriptive and not row.effect_supported
    assert not row.interval_available and row.ci_low == 1.5


def test_unknown_source_support_keeps_declared_interval_without_claiming_supported_inference(
    data,
):
    shifts, _ = reports()
    shifts = shifts.drop(columns=["n_cells0", "n_cells1", "n_samples0", "n_samples1"])
    result = plot(
        data,
        mode="stored",
        shift_df=shifts,
        analysis_rep="X_pca",
        effect_metric="delta_norm",
        layout="summary",
    )
    row = result.data["evidence"].set_index("state").loc["Shared"]
    assert row.effect_available and row.effect_descriptive and not row.effect_supported
    assert row.effect_support_status == "source_support_not_recorded"
    assert (
        row.interval_available
        and row.interval_reason == "stored_interval_unverified_support"
    )
    assert any("unverified source support" in warning for warning in result.warnings)
    assert any(
        "source support unverified" in text.get_text()
        for text in result.axes["effect"].texts
    )


@pytest.mark.parametrize("key", ["velocity_norm", "vel_norm"])
def test_zero_recorded_velocity_makes_stored_cosine_unavailable(data, key):
    shifts, alignment = reports()
    alignment[key] = 1.0
    alignment.loc[alignment.node.eq("Shared"), key] = 0.0
    result = plot(
        data,
        mode="stored",
        shift_df=shifts,
        alignment_df=alignment,
        analysis_rep="X_pca",
    )
    row = result.data["evidence"].set_index("state").loc["Shared"]
    assert row.alignment == 0.7 and not row.alignment_available
    assert row.alignment_reason == "undefined_zero_velocity"
