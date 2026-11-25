"""PAGA composition contracts with sparse conditions and unequal capture."""

from __future__ import annotations

import copy
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
def _close_figures():
    yield
    plt.close("all")


def _data():
    records, coordinates = [], []
    for sample, counts in {
        "C1": (10, 0),
        "C2": (19, 1),
        "T1": (4, 6),
        "T2": (5, 15),
    }.items():
        condition = "control" if sample.startswith("C") else "treated"
        for state, count in zip(("Shared", "Enriched"), counts):
            base = np.array([0.0, 0.0]) if state == "Shared" else np.array([5.0, 2.0])
            base += [1.0, 0.5] if condition == "treated" else [0.0, 0.0]
            for i in range(count):
                records.append(
                    {"state": state, "condition": condition, "sample": sample}
                )
                coordinates.append(base + [i * 0.01, i * 0.002])
    obs = pd.DataFrame(records, index=[f"synthetic_{i}" for i in range(len(records))])
    obs["state"] = pd.Categorical(
        obs.state, categories=["Shared", "Enriched", "Absent"]
    )
    obs["condition"] = pd.Categorical(obs.condition, categories=["control", "treated"])
    adata = AnnData(np.zeros((len(obs), 1)), obs=obs)
    adata.obsm["X_umap"] = np.asarray(coordinates)
    adata.obsm["velocity_umap"] = np.tile([0.2, 0.1], (len(obs), 1))
    adata.uns["paga"] = {
        "groups": "state",
        "node_order": ["Shared", "Enriched", "Absent"],
        "connectivities": sparse.csr_matrix(
            [[0, 0.8, 0.4], [0.8, 0, 0.1], [0.4, 0.1, 0]]
        ),
    }
    return adata


def _plot(adata, **kwargs):
    options = {
        "node_key": "state",
        "condition_key": "condition",
        "group0": "control",
        "group1": "treated",
        "min_cells": 3,
        "label": False,
        "legend": False,
        "show": False,
        "return_result": True,
    }
    options.update(kwargs)
    return sg.pl.paga_shift_map(adata, **options)


def _pooled(adata, **kwargs):
    options = {
        "node_position": "pooled",
        "show_shift": False,
        "show_endpoints": False,
        "pie_key": "condition",
    }
    options.update(kwargs)
    return _plot(adata, **options)


def _balanced(adata, **kwargs):
    return _pooled(
        adata, pie_mode="sample_abundance", pie_sample_key="sample", **kwargs
    )


def _copy_rows(adata, indices, *, sample=None, state=None):
    extra = adata.obs.iloc[indices].copy()
    if sample is not None:
        extra["sample"] = sample
    if state is not None:
        extra["state"] = state
    obs = pd.concat([adata.obs, extra], ignore_index=True)
    obs.index = [f"copy_{i}" for i in range(len(obs))]
    result = AnnData(np.zeros((len(obs), 1)), obs=obs)
    for key, values in adata.obsm.items():
        result.obsm[key] = np.vstack([values, values[indices]])
    result.uns = copy.deepcopy(adata.uns)
    return result


def test_default_reference_and_raw_composition_contract_is_unchanged():
    adata = _data()
    default = _plot(adata, pie_key="condition")
    explicit = _plot(
        adata,
        pie_key="condition",
        node_position="reference",
        show_shift=True,
        pie_mode="cell_fraction",
    )
    for key in (
        "centroids",
        "vectors",
        "edge_table",
        "composition",
        "composition_counts",
    ):
        pd.testing.assert_frame_equal(default.data[key], explicit.data[key])
    assert "Enriched" not in default.artists["pies"]
    fractions = default.data["composition"].set_index("node")
    assert fractions.loc["Enriched", "control"] == pytest.approx(1 / 22)
    values = default.data["pie_values"]
    assert set(values["mode"]) == {"cell_fraction"}
    np.testing.assert_allclose(values.fraction, values.cell_fraction)


def test_pooled_pies_keep_sparse_reference_nodes_and_supplied_edges():
    result = _pooled(_data())
    anchors = result.data["node_anchors"].set_index("state")
    centers = result.data["centroids"].set_index("state")
    assert centers.loc["Enriched", "n0"] == 1
    assert not centers.loc["Enriched", "present0"]
    assert anchors.loc["Enriched", "n_cells"] == 22
    assert anchors.loc["Enriched", "drawn"] and anchors.loc["Enriched", "pie_drawn"]
    assert "Enriched" in result.artists["pies"]
    assert not anchors.loc["Absent", "observed"]
    edges = result.data["edge_table"].set_index(["source", "target"])
    assert edges.loc[("Shared", "Enriched"), "displayed"]
    assert not edges.loc[("Shared", "Absent"), "displayed"]
    assert edges.loc[("Shared", "Absent"), "reason"] == "missing_pooled_anchor"
    assert result.artists["vectors"]["shift"] == []
    assert result.artists["comparison_nodes"] == {}
    assert result.artists["reference_nodes"] == {}
    assert result.data["vectors"].reason.eq("shift_disabled").all()
    assert result.metadata["graph_directed"] is False
    assert anchors.loc["Shared", "size_points2"] == pytest.approx(220)
    assert anchors.loc["Enriched", "size_points2"] == pytest.approx(
        220 * (0.55 + 0.45 * np.sqrt(22 / 38))
    )


def test_pooled_node_survives_fully_absent_reference_and_low_total_coverage():
    adata = _data()
    keep = ~(adata.obs.state.eq("Enriched") & adata.obs.condition.eq("control"))
    adata = adata[keep.to_numpy()].copy()
    result = _pooled(adata, min_cells=100)
    anchor = result.data["node_anchors"].set_index("state").loc["Enriched"]
    assert anchor.low_coverage and anchor.drawn and anchor.pie_drawn
    center = result.data["centroids"].set_index("state").loc["Enriched"]
    assert center.coverage_status == "comparison_only"
    assert np.isnan(center.x0)
    assert "Enriched" in result.artists["pies"]


def test_pooled_graph_positions_do_not_reanchor_actual_shift_or_velocity():
    adata = _data()
    reference = _plot(adata, min_cells=1, show_velocity=True)
    pooled = _plot(adata, node_position="pooled", min_cells=1, show_velocity=True)
    pd.testing.assert_frame_equal(reference.data["centroids"], pooled.data["centroids"])
    pd.testing.assert_frame_equal(reference.data["vectors"], pooled.data["vectors"])
    pd.testing.assert_frame_equal(reference.data["velocity"], pooled.data["velocity"])
    anchors = pooled.data["node_anchors"].set_index("state")
    centers = pooled.data["centroids"].set_index("state")
    assert anchors.loc["Enriched", "x"] != pytest.approx(centers.loc["Enriched", "x0"])
    vectors = pooled.data["vectors"].set_index(["state", "kind"])
    for kind in ("shift", "velocity"):
        assert (
            vectors.loc[("Enriched", kind), "source_x"] == centers.loc["Enriched", "x0"]
        )
    edges = pooled.data["edge_table"].set_index(["source", "target"])
    assert edges.loc[("Shared", "Enriched"), "target_x"] == anchors.loc["Enriched", "x"]


def test_sample_abundance_uses_all_samples_including_zero_state_counts():
    result = _balanced(_data())
    sample = result.data["pie_sample_counts"].set_index(
        ["state", "condition", "sample"]
    )
    zero = sample.loc[("Enriched", "control", "C1")]
    assert zero["count"] == 0 and zero.denominator == 10
    assert zero.state_fraction == 0 and zero.included
    values = result.data["pie_values"].set_index(["state", "category_label"])
    control = values.loc[("Enriched", "control")]
    treated = values.loc[("Enriched", "treated")]
    assert control.mean_sample_abundance == pytest.approx((0 + 1 / 20) / 2)
    assert treated.mean_sample_abundance == pytest.approx((6 / 10 + 15 / 20) / 2)
    assert control.fraction == pytest.approx(1 / 28)
    assert treated.fraction == pytest.approx(27 / 28)
    assert control.cell_fraction == pytest.approx(1 / 22)
    assert control.n_samples == control.n_samples_total == 2
    raw = result.data["composition_counts"].set_index(["state", "category_label"])
    assert raw.loc[("Enriched", "control"), "fraction"] == pytest.approx(1 / 22)
    assert result.data["composition"].set_index("node").loc[
        "Enriched", "control"
    ] == pytest.approx(1 / 28)


def test_condition_means_do_not_weight_conditions_by_sample_roster_size():
    adata = _data()
    extra = np.flatnonzero(adata.obs["sample"].eq("C1"))
    enlarged = _copy_rows(adata, extra, sample="C3")
    result = _balanced(enlarged)
    values = result.data["pie_values"].set_index(["state", "category_label"])
    control = values.loc[("Enriched", "control")]
    treated = values.loc[("Enriched", "treated")]
    assert control.n_samples == 3 and treated.n_samples == 2
    assert control.mean_sample_abundance == pytest.approx(1 / 60)
    assert treated.mean_sample_abundance == pytest.approx(27 / 40)
    assert control.fraction == pytest.approx(2 / 83)
    assert treated.fraction == pytest.approx(81 / 83)


def test_whole_sample_capture_duplication_changes_raw_pies_but_not_balanced_pies():
    adata = _data()
    extra = np.tile(np.flatnonzero(adata.obs["sample"].eq("T2")), 4)
    repeated = _copy_rows(adata, extra)
    base, copied = _balanced(adata), _balanced(repeated)
    pd.testing.assert_frame_equal(base.data["composition"], copied.data["composition"])
    raw0 = base.data["composition_counts"].set_index(["state", "category_label"])
    raw1 = copied.data["composition_counts"].set_index(["state", "category_label"])
    assert (
        raw0.loc[("Enriched", "control"), "fraction"]
        != raw1.loc[("Enriched", "control"), "fraction"]
    )
    sample = copied.data["pie_sample_counts"].set_index(
        ["state", "condition", "sample"]
    )
    assert sample.loc[("Enriched", "treated", "T2"), "denominator"] == 100
    assert sample.loc[("Enriched", "treated", "T2"), "state_fraction"] == pytest.approx(
        0.75
    )


def test_reused_paired_ids_remain_separate_sample_condition_units():
    adata = _data()
    original = _balanced(adata)
    adata.obs["sample"] = adata.obs["sample"].replace({"T1": "C1", "T2": "C2"})
    paired = _balanced(adata)
    pd.testing.assert_frame_equal(
        original.data["composition"], paired.data["composition"]
    )
    units = paired.data["pie_sample_counts"][["condition", "sample"]].drop_duplicates()
    assert len(units) == 4 and units["sample"].nunique() == 2


def test_missing_annotation_denominator_is_unavailable_not_zero_abundance():
    adata = _data()
    extra = np.array([0, 1, 2])
    enlarged = _copy_rows(adata, extra, sample="C_missing")
    enlarged.obs.loc[enlarged.obs["sample"].eq("C_missing"), "state"] = np.nan
    result = _balanced(enlarged)
    units = result.data["pie_sample_counts"]
    unavailable = units.loc[units["sample"].eq("C_missing")]
    assert unavailable.denominator.eq(0).all()
    assert unavailable.state_fraction.isna().all()
    assert not unavailable.included.any()
    values = result.data["pie_values"].set_index(["state", "category_label"])
    control = values.loc[("Enriched", "control")]
    assert control.n_samples == 2 and control.n_samples_total == 3
    assert control.n_samples_missing_denominator == 1
    assert control.fraction == pytest.approx(1 / 28)
    assert any("no annotated cells" in note for note in result.warnings)


def test_whole_condition_missing_annotation_does_not_become_a_full_other_slice():
    adata = _data()
    adata.obs.loc[adata.obs.condition.eq("control"), "state"] = np.nan
    result = _balanced(adata)
    assert result.metadata["pie_categories"] == ["control", "treated"]
    assert set(result.data["pie_values"].category_label) == {"control", "treated"}
    control = result.data["pie_values"].loc[
        result.data["pie_values"].category_label.eq("control")
    ]
    assert control.mean_sample_abundance.isna().all()
    assert control.n_samples.eq(0).all()
    assert control.n_samples_total.eq(2).all()
    assert result.data["pie_values"].fraction.isna().all()
    assert result.data["pie_values"].status.eq("missing_sample_denominator").all()
    assert result.artists["pies"] == {}
    assert result.data["node_anchors"].set_index("state").loc["Enriched", "drawn"]


def test_sample_denominators_ignore_focus_and_display_subsampling():
    result = _balanced(
        _data(), focus_states=["Enriched"], edge_focus="incident", max_points=4
    )
    row = (
        result.data["pie_sample_counts"]
        .set_index(["state", "condition", "sample"])
        .loc[("Enriched", "control", "C1")]
    )
    assert row.denominator == 10 and row["count"] == 0
    assert result.data["display_cells"].displayed.sum() == 4


def test_pooled_category_palette_and_slice_order_survive_cell_reordering():
    adata = _data()
    adata.uns["condition_colors"] = ["#4477AA", "#D67732"]
    first = _pooled(adata)
    second = _pooled(adata[np.arange(adata.n_obs)[::-1]].copy())
    assert first.metadata["pie_categories"] == second.metadata["pie_categories"]
    assert first.metadata["pie_palette"] == second.metadata["pie_palette"]
    pd.testing.assert_frame_equal(first.data["composition"], second.data["composition"])


def test_unused_categorical_label_collision_fails_before_drawing():
    adata = _data()
    adata.obs["annotation"] = pd.Categorical([1] * adata.n_obs, categories=[1, "1"])
    _, ax = plt.subplots()
    before = list(ax.get_children())
    with pytest.raises(ValueError, match="identical string labels"):
        _pooled(adata, pie_key="annotation", ax=ax)
    assert list(ax.get_children()) == before


def test_explicit_categories_cannot_hide_selected_condition_without_denominator():
    adata = _data()
    adata.obs.loc[adata.obs.condition.eq("control"), "state"] = np.nan
    _, ax = plt.subplots()
    before = list(ax.get_children())
    with pytest.raises(ValueError, match="both selected conditions"):
        _balanced(adata, pie_categories=["treated"], ax=ax)
    assert list(ax.get_children()) == before


def test_missing_slice_cannot_duplicate_unused_literal_missing_category():
    adata = _data()
    labels = ["known"] * adata.n_obs
    labels[-1] = None
    adata.obs["annotation"] = pd.Categorical(labels, categories=["known", "(missing)"])
    result = _pooled(adata, pie_key="annotation")
    categories = result.metadata["pie_categories"]
    assert categories == ["known", "((missing))"]
    counts = result.data["composition_counts"]
    enriched = counts.loc[counts.state.eq("Enriched")]
    assert len(enriched) == 2
    assert enriched["count"].sum() == 22
    assert enriched.fraction.sum() == pytest.approx(1.0)
    missing = enriched.loc[enriched.is_missing].iloc[0]
    assert missing["count"] == 1
    assert pd.isna(missing.category)


@pytest.mark.parametrize(
    "options",
    [
        {"node_position": "unknown"},
        {"show_shift": "false"},
        {"pie_mode": "unknown"},
        {
            "pie_mode": "sample_abundance",
            "pie_key": "sample",
            "pie_sample_key": "sample",
        },
        {
            "pie_mode": "sample_abundance",
            "pie_key": "condition",
            "pie_sample_key": "absent",
        },
        {"pie_sample_key": "sample"},
    ],
)
def test_invalid_modes_fail_before_axes_changes(options):
    _, ax = plt.subplots()
    before = list(ax.get_children())
    with pytest.raises((ValueError, KeyError, TypeError)):
        _plot(_data(), ax=ax, **options)
    assert list(ax.get_children()) == before


def test_missing_sample_label_fails_before_drawing():
    adata = _data()
    adata.obs.loc[adata.obs_names[0], "sample"] = None
    _, ax = plt.subplots()
    before = list(ax.get_children())
    with pytest.raises(ValueError, match="missing sample"):
        _balanced(adata, ax=ax)
    assert list(ax.get_children()) == before


def test_pie_export_preserves_sample_denominators_and_anchor_source(tmp_path):
    result = _balanced(_data())
    paths = result.save(tmp_path / "pies", formats=("png",), dpi=40)
    metadata = json.loads(Path(paths["metadata"]).read_text())
    for key in (
        "node_anchors",
        "pie_values",
        "pie_sample_counts",
        "composition_counts",
    ):
        table_path = tmp_path / metadata["tables"][key]["file"]
        assert table_path.is_file()
        assert len(pd.read_csv(table_path)) == len(result.data[key])
    assert (
        result.metadata["pie_sample_denominator"]
        == "annotated cells per sample-condition unit"
    )
    assert result.metadata["node_position"] == "pooled"
    assert result.metadata["show_shift"] is False
