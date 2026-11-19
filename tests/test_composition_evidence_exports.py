"""Stored composition and volcano evidence must never be filled or refitted."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import scgeo as sg


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def data():
    frame = pd.DataFrame({"node": ["A", "B", "C", "D", "E"],
                          "effect": [2., np.nan, -1., 0., 3.],
                          "q": [0., .02, np.nan, 1., .1]},
                         index=pd.Index(["r1", "r1", "r2", "r3", "r4"], name="source_id"))
    frame.attrs["provenance"] = {"example": "synthetic stored table"}
    result = AnnData(obs=pd.DataFrame(index=["cell"]))
    result.uns["scgeo"] = {"paga_composition_stats": {"table": frame,
        "params": {"method": "bootstrap", "group0": "A", "group1": "B", "sample_key": "sample"}},
        "distribution_test": frame.copy(deep=True)}
    return result


def test_composition_volcano_omits_missing_without_zero_or_one_fill(data):
    before = data.uns["scgeo"]["paga_composition_stats"]["table"].copy(deep=True)
    result = sg.pl.paga_composition_volcano(data, return_result=True, show=False)
    nodes = result.data["nodes"]
    assert nodes.displayed.tolist() == [True, False, False, True, True]
    assert np.isnan(nodes.loc[1, "effect"])
    assert np.isnan(nodes.loc[2, "p_value"])
    assert nodes.loc[0, "p_value"] == 0 and nodes.loc[0, "neglog10_p"] == 300
    assert nodes.loc[0, "p_clipped_for_display"]
    assert not nodes.loc[1:2, "highlighted"].any()
    assert len(result.artists["points"].get_offsets()) == 3
    pd.testing.assert_frame_equal(result.data["source"], before)
    pd.testing.assert_frame_equal(data.uns["scgeo"]["paga_composition_stats"]["table"], before)
    assert result.metadata["analysis_params"]["sample_key"] == "sample"


def test_composition_bar_marks_unavailable_effects_and_probabilities(data):
    result = sg.pl.paga_composition_bar(data, top_k=5, return_result=True, show=False)
    assert len(result.artists["bars"]) == 4
    nodes = result.data["nodes"]
    assert nodes.selected.all()
    assert nodes.displayed.sum() == 4
    assert np.isnan(nodes.loc[1, "effect"]) and np.isnan(nodes.loc[2, "p_value"])
    text = " ".join(t.get_text() for t in result.artists["annotations"])
    assert "effect unavailable" in text and "q unavailable" in text
    assert result.data["rows"].iloc[-1].node == "A"  # strongest finite p at top


def test_absent_p_column_is_explicit_effect_only_strip(data):
    del data.uns["scgeo"]["paga_composition_stats"]["table"]["q"]
    result = sg.pl.paga_composition_volcano(data, return_result=True, show=False)
    assert result.metadata["ordinate_mode"] == "effect_only"
    assert result.data["nodes"].p_value.isna().all()
    assert result.data["nodes"].neglog10_p.isna().all()
    assert "no p/q" in result.axes["volcano"].get_ylabel()
    assert result.data["nodes"].displayed.sum() == 4


def test_an_entirely_missing_p_column_stays_unavailable(data):
    data.uns["scgeo"]["paga_composition_stats"]["table"]["q"] = np.nan
    result = sg.pl.paga_composition_volcano(data, return_result=True, show=False)
    assert not result.data["nodes"].displayed.any()
    assert not result.data["nodes"].highlighted.any()
    assert len(result.artists["points"].get_offsets()) == 0


@pytest.mark.parametrize("invalid", [-.01, 1.01, np.inf])
def test_invalid_probabilities_rejected_before_any_drawing(data, invalid):
    data.uns["scgeo"]["paga_composition_stats"]["table"].iloc[0, 2] = invalid
    for function in (sg.pl.paga_composition_bar, sg.pl.paga_composition_volcano,
                     sg.pl.paga_composition_panel):
        with pytest.raises(ValueError, match="probabilities|finite"):
            function(data, show=False)
    assert not plt.get_fignums()


def test_stored_odds_ratios_remain_on_original_scale(data):
    data.uns["scgeo"]["paga_composition_stats"]["table"] = pd.DataFrame({
        "node": ["A", "B"], "OR": [.5, 2.], "q": [.04, .01],
        "CI_low": [.2, 1.1], "CI_high": [.8, 2.9], "method": ["gee", "gee"]})
    result = sg.pl.paga_composition_bar(data, return_result=True, show=False)
    assert result.metadata["effect_col"] == "OR" and result.metadata["null_effect"] == 1
    assert result.data["nodes"].effect.tolist() == [.5, 2.]
    assert result.axes["bar"].get_xlabel() == "OR"
    assert all(patch.get_x() == 1 for patch in result.artists["bars"])
    # Interval values are retained but not relabelled as log-odds intervals.
    assert result.data["source"].CI_low.tolist() == [.2, 1.1]


def test_composition_subfigure_and_complete_source_exports(data, tmp_path):
    fig = plt.figure(figsize=(10, 5))
    left, right = fig.subfigures(1, 2)
    axs = left.subplots(1, 2)
    sibling = right.subplots()
    positions = [ax.get_position(original=True).bounds for ax in axs]
    result = sg.pl.paga_composition_panel(
        data, axes={"volcano": axs[0], "bar": axs[1]}, return_result=True, show=False)
    assert result.fig is fig and len(fig.axes) == 3
    np.testing.assert_allclose([ax.get_position(original=True).bounds for ax in axs], positions)
    output = result.save(tmp_path / "composition", formats=("png", "svg", "pdf"), dpi=60)
    manifest = json.loads(Path(output["metadata"]).read_text())
    assert manifest["plot"]["export_scope"] == "parent_figure"
    assert manifest["tables"]["volcano_source"]["rows"] == 5
    assert manifest["tables"]["bar_source"]["attributes"]["provenance"]["example"] == "synthetic stored table"
    loaded = pd.read_csv(output["tables"]["bar_source"])
    assert loaded.effect.isna().sum() == loaded.q.isna().sum() == 1
    assert loaded.source_id.tolist() == ["r1", "r1", "r2", "r3", "r4"]


def test_distribution_volcano_sources_and_legacy_return(data):
    result = sg.pl.distribution_test_volcano(
        data, effect_col="effect", p_col="q", label_col="node", p_threshold=None,
        return_result=True, show=False)
    assert result.data["points"].displayed.tolist() == [True, False, False, True, True]
    assert result.data["points"].labelled.sum() == 3
    assert "threshold" not in result.artists
    fig, ax, source = sg.pl.distribution_test_volcano(data, p_col="q", show=False)
    pd.testing.assert_frame_equal(source, result.data["source"])
    data.uns["scgeo"]["distribution_test"].iloc[0, 2] = -.1
    with pytest.raises(ValueError, match="probabilities"):
        sg.pl.distribution_test_volcano(data, p_col="q", show=False)


def test_nullable_values_and_zero_selection(data):
    table = data.uns["scgeo"]["paga_composition_stats"]["table"]
    table["effect"] = table.effect.astype("Float64")
    table["q"] = table.q.astype("Float64")
    fig, axes, tables = sg.pl.paga_composition_bar(data, top_k=0, return_data=True, show=False)
    assert tables["rows"].empty and not tables["nodes"].selected.any()
    assert len(axes) == 1
