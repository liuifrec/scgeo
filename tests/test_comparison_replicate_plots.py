import copy

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scgeo.get._sample_centers import sample_centers
from scgeo.pl._evidence_comparison import evidence_comparison
from scgeo.pl._replicate_effect import replicate_effect
from scgeo.tl._robust_shift import robust_shift


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _sample_adata():
    # Unequal cell counts must not become unequal replicate weights.
    obs, values = [], []
    for condition, sample, n, x in [("A", "a1", 2, 0.), ("A", "a2", 9, 4.),
                                     ("B", "b1", 3, 2.), ("B", "b2", 12, 6.)]:
        for i in range(n):
            obs.append({"condition": condition, "sample": sample, "state": "T" if i else "B"})
            values.append([x, x+1])
    obs = pd.DataFrame(obs, index=[f"cell_{i}" for i in range(len(obs))])
    data = ad.AnnData(np.zeros((len(obs), 1)), obs=obs)
    data.obsm["X_pca"] = np.asarray(values)
    robust_shift(data, sample_key="sample", by="state", center="mean", group0="A", group1="B", n_boot=16, seed=7)
    return data


def _effects():
    return pd.DataFrame({
        "state": ["T", "B", "T"], "contrast": ["D8 vs D0", "D8 vs D0", "D28 vs D0"],
        "metric": ["delta_norm"]*3, "estimate": [.07, .8, .4],
        "interval_low": [.12, .6, .2], "interval_high": [.28, 1., .5],
        "interval_type": ["sample_bootstrap_ci95", "sample_bootstrap_ci95", "representation_iqr"],
        "representation": ["X_pca"]*3,
        "unit": ["Euclidean distance"]*3,
    })


def test_stored_sample_centers_are_exact_replicates_and_getter_is_immutable():
    data = _sample_adata()
    block = data.uns["scgeo"]["robust_shift"]
    np.testing.assert_array_equal(block["global"]["sample_centers0"], [[0, 1], [4, 5]])
    np.testing.assert_array_equal(block["global"]["sample_centers1"], [[2, 3], [6, 7]])
    assert block["global"]["delta_norm"] == pytest.approx(np.sqrt(8))
    table = sample_centers(data)
    assert table.shape[0] == 2*4*2  # states x samples x coordinates
    assert set(table.state) == {"T", "B"}
    assert table.attrs["coordinate_indexing"] == "zero-based"
    assert table.attrs["provenance"]["resolved_bootstrap_unit"] == "sample"
    table.loc[:, "value"] = -100
    table.attrs["provenance"]["center"] = "changed"
    np.testing.assert_array_equal(block["global"]["sample_centers0"], [[0, 1], [4, 5]])
    assert block["params"]["center"] == "mean"


def test_sample_centers_h5ad_round_trip(tmp_path):
    data = _sample_adata()
    original = sample_centers(data)
    target = tmp_path / "samples.h5ad"
    data.write_h5ad(target)
    loaded = ad.read_h5ad(target)
    pd.testing.assert_frame_equal(sample_centers(loaded), original)


def test_sample_centers_available_when_state_missing_in_other_condition():
    data = _sample_adata()
    data.obs["state"] = np.where(data.obs.condition == "A", "A only", "B only")
    robust_shift(data, sample_key="sample", by="state", group0="A", group1="B", n_boot=4)
    table = sample_centers(data)
    assert set(table.state) == {"A only", "B only"}
    assert len(table) == 8
    assert np.isnan(data.uns["scgeo"]["robust_shift"]["by"]["A only"]["delta_norm"])


def test_old_or_cell_level_results_do_not_fabricate_replicates():
    data = _sample_adata()
    old = copy.deepcopy(data.uns["scgeo"]["robust_shift"])
    del old["by"]["T"]["sample_centers0"]
    with pytest.raises(ValueError, match="predates stored sample centers"):
        sample_centers(old)
    robust_shift(data, group0="A", group1="B", n_boot=4)
    with pytest.raises(ValueError, match="Cell-level results"):
        sample_centers(data)


def test_evidence_comparison_preserves_intervals_missing_states_and_input():
    table = _effects()
    before = table.copy(deep=True)
    result = evidence_comparison(table, state_order=["B", "T", "not sampled"], return_result=True, show=False)
    assert list(result.axes) == ["D8 vs D0", "D28 vs D0"]
    intervals = result.artists["D8 vs D0"]["intervals"]
    np.testing.assert_allclose(intervals[1].get_segments()[0][:, 0], [.12, .28])
    assert result.data.shape[0] == 6
    assert result.data.availability.eq("not_computed").sum() == 3
    assert result.metadata["shared_limits"]
    assert set(result.metadata["interval_types"]) == {"sample_bootstrap_ci95", "representation_iqr"}
    assert result.axes["D8 vs D0"].get_xlim() == result.axes["D28 vs D0"].get_xlim()
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("case,match", [
    ("duplicates", "Duplicate"), ("interval_type", "interval_type"),
    ("reversed", "must not exceed"), ("one_endpoint", "two endpoints"),
    ("multiple_metrics", "Select metric"),
])
def test_evidence_comparison_rejects_ambiguous_evidence(case, match):
    table = _effects()
    if case == "duplicates":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    elif case == "interval_type":
        table.loc[0, "interval_type"] = None
    elif case == "reversed":
        table.loc[0, "interval_low"] = 2
    elif case == "one_endpoint":
        table.loc[0, "interval_high"] = np.nan
    else:
        table.loc[0, "metric"] = "another_metric"
    with pytest.raises(ValueError, match=match):
        evidence_comparison(table, show=False)


def test_evidence_comparison_does_not_share_unrelated_representation_scales():
    table = _effects()
    table.loc[table.contrast == "D28 vs D0", "representation"] = "X_other"
    result = evidence_comparison(table, return_result=True, show=False)
    assert not result.metadata["shared_limits"]
    with pytest.raises(ValueError, match="Shared limits"):
        evidence_comparison(table, sharex=True, show=False)
    table["scale_id"] = "explicitly calibrated common scale"
    result = evidence_comparison(table, sharex=True, return_result=True, show=False)
    assert result.metadata["shared_limits"]


def test_comparison_requires_declared_units_and_preserves_supplied_status():
    table = _effects().drop(columns="unit")
    result = evidence_comparison(table, return_result=True, show=False)
    assert not result.metadata["shared_limits"]
    with pytest.raises(ValueError, match="Shared limits"):
        evidence_comparison(table, sharex=True, show=False)
    table["evidence_status"] = ["insufficient_coverage", "assessed", "unavailable"]
    result = evidence_comparison(table, return_result=True, show=False)
    plotted = result.data.set_index(["contrast", "state"])
    assert plotted.loc[("D8 vs D0", "T"), "availability"] == "insufficient_coverage"
    assert plotted.loc[("D28 vs D0", "T"), "display_status"] == "unavailable"
    labels = [text.get_text() for text in result.axes["D8 vs D0"].get_legend().get_texts()]
    assert "insufficient coverage" in labels
    table["scale_id"] = ""
    table.loc[1, "representation"] = "X_other"
    with pytest.raises(ValueError, match="mixes representations"):
        evidence_comparison(table, show=False)


def test_comparison_supplied_axes_and_source_export(tmp_path):
    table = _effects()
    fig, axs = plt.subplots(1, 2)
    figures_before = plt.get_fignums()
    result = evidence_comparison(table, axes=dict(zip(table.contrast.unique(), axs)), return_result=True, show=False)
    assert result.fig is fig
    assert plt.get_fignums() == figures_before
    result.save(tmp_path / "comparison", formats=("svg",), include_data=True, include_metadata=True)
    assert list(tmp_path.glob("*.svg"))
    assert list(tmp_path.rglob("*.csv"))
    assert list(tmp_path.rglob("*.json"))


def test_replicate_plot_uses_samples_and_equal_weight_summary():
    data = _sample_adata()
    result = replicate_effect(data, scope="global", coordinate=0, return_result=True, show=False)
    samples = result.data["sample_centers"]
    assert len(samples) == 4
    assert samples.cell_count.sum() == data.n_obs
    summary = result.data["sample_summary"].set_index("condition")
    assert summary.loc["A", "value"] == 2
    assert summary.loc["B", "value"] == 4
    assert set(summary.n_samples) == {2}
    assert result.data["pairs"].empty
    assert result.metadata["pair_key"] is None


def test_replicate_pairs_require_explicit_metadata_and_are_not_inferred():
    table = sample_centers(_sample_adata(), scope="global")
    table["donor"] = table["sample"].str[-1]
    without = replicate_effect(table, coordinate=0, return_result=True, show=False)
    assert without.data["pairs"].empty
    result = replicate_effect(table, coordinate=0, pair_key="donor", return_result=True, show=False)
    assert len(result.data["pairs"]) == 2
    assert set(result.data["pairs"].sample0) == {"a1", "a2"}
    assert set(result.data["pairs"].sample1) == {"b1", "b2"}
    table["donor"] = "ambiguous"
    with pytest.raises(ValueError, match="at most one sample"):
        replicate_effect(table, coordinate=0, pair_key="donor", show=False)


def test_replicate_validation_and_supplied_axes():
    table = sample_centers(_sample_adata())
    fig, axs = plt.subplots(1, 2)
    figures_before = plt.get_fignums()
    result = replicate_effect(table, coordinate=0, axes={"T": axs[0], "B": axs[1]}, return_result=True, show=False)
    assert result.fig is fig
    assert plt.get_fignums() == figures_before
    with pytest.raises(ValueError, match="No sample centers"):
        replicate_effect(table, coordinate=99, show=False)
    duplicate = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one selected"):
        replicate_effect(duplicate, coordinate=0, show=False)
    table.loc[0, "representation"] = "X_other"
    with pytest.raises(ValueError, match="one representation"):
        replicate_effect(table, coordinate=0, show=False)


def test_replicate_missing_values_stay_visible():
    table = sample_centers(_sample_adata(), scope="global")
    table.loc[(table["sample"] == "a1") & (table.coordinate == 0), "value"] = np.nan
    result = replicate_effect(table, coordinate=0, return_result=True, show=False)
    summary = result.data["sample_summary"].set_index("condition")
    assert summary.loc["A", "n_samples"] == 1
    assert summary.loc["A", "n_missing"] == 1
    assert len(result.data["sample_centers"]) == 4


def test_many_sample_states_wrap_without_changing_numeric_sources():
    table = pd.DataFrame([
        dict(state=f"State {i} long biological description", condition=group,
             sample=f"{group}_{j}", coordinate=0, value=float(i + j),
             representation="X_pca")
        for i in range(6) for group in ("control", "treated") for j in range(3)
    ])
    result = replicate_effect(table, coordinate=0, return_result=True, show=False)
    explicit = replicate_effect(table, coordinate=0, figsize=(20, 4), return_result=True, show=False)
    assert result.metadata["resolved_layout"] == "wrapped"
    axes = list(result.axes.values())
    assert axes[0].get_position().y0 > axes[3].get_position().y0
    assert axes[3].get_ylabel()
    assert axes[0].get_ylim() == axes[-1].get_ylim()
    pd.testing.assert_frame_equal(result.data["sample_centers"], explicit.data["sample_centers"])
    pd.testing.assert_frame_equal(result.data["sample_summary"], explicit.data["sample_summary"])
