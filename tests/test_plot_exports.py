"""Export provenance, scoped styling, and display-only selection guarantees."""
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scgeo.pl import PlotResult, category_palette, export_figure, style_context
from scgeo.pl._utils import display_indices, resolve_embedding


def test_export_roundtrips_source_values_and_hashes(tmp_path):
    table = pd.DataFrame({"estimate": [0.07, np.nan], "low": [0.12, np.nan],
                          "high": [0.28, np.nan]}, index=pd.Index(["A", "B"], name="state"))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter([0.07], [0])
    result = PlotResult(fig, {"effect": ax}, table, {"metric": "raw magnitude", "seed": np.int64(4)},
                        ["Synthetic example"])
    old_size = fig.get_size_inches().copy()
    output = result.save(tmp_path / "figure.svg", formats=("svg", "png", "pdf"),
                         width_mm=101.6, height_mm=76.2, panel_labels={"effect": "A"}, dpi=100)
    manifest = json.loads((tmp_path / "figure.json").read_text())
    assert manifest["export"]["canvas_mm"] == pytest.approx([101.6, 76.2])
    assert manifest["plot"]["seed"] == 4
    assert manifest["export"]["panel_labels"] == {"effect": "A"}
    assert manifest["warnings"] == ["Synthetic example"]
    loaded = pd.read_csv(output["tables"]["data"], index_col="state")
    pd.testing.assert_frame_equal(loaded, table)
    for fmt, path in output["figures"].items():
        from pathlib import Path
        assert manifest["figures"][fmt]["sha256"] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert manifest["tables"]["data"]["sha256"] == hashlib.sha256((tmp_path / "figure_data.csv").read_bytes()).hexdigest()
    assert len(ax.texts) == 0
    np.testing.assert_array_equal(fig.get_size_inches(), old_size)
    plt.close(fig)


def test_export_restores_canvas_and_labels_after_save_error(tmp_path, monkeypatch):
    fig, ax = plt.subplots(figsize=(4, 3))
    old_size = fig.get_size_inches().copy()
    def fail(*args, **kwargs):
        raise OSError("disk unavailable")
    monkeypatch.setattr(fig, "savefig", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        export_figure(PlotResult(fig, {"main": ax}), tmp_path / "broken", width_mm=140,
                      panel_labels={"main": "A"})
    np.testing.assert_array_equal(fig.get_size_inches(), old_size)
    assert not ax.texts
    plt.close(fig)


def test_export_preserves_nondefault_unnamed_range_index(tmp_path):
    table = pd.DataFrame({"estimate": [.1, .2, .3]}, index=pd.RangeIndex(10, 16, 2))
    fig, ax = plt.subplots()
    output = PlotResult(fig, {"effect": ax}, table).save(tmp_path / "range_index", formats="png")
    loaded = pd.read_csv(output["tables"]["data"], index_col=0)
    assert loaded.index.tolist() == [10, 12, 14]
    assert json.loads((tmp_path / "range_index.json").read_text())["tables"]["data"]["index_included"]
    plt.close(fig)


def test_export_rejects_invalid_metadata_and_table_collisions_before_writes(tmp_path):
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="metadata type"):
        export_figure(PlotResult(fig, {"main": ax}, metadata={"axis": ax}), tmp_path / "invalid")
    frame = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="collide"):
        export_figure(PlotResult(fig, {"main": ax}, {"a/b": frame, "a_b": frame}), tmp_path / "invalid")
    assert not list(tmp_path.iterdir())
    plt.close(fig)


def test_svg_reexport_is_deterministic(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    export_figure(fig, tmp_path / "first", formats="svg")
    export_figure(fig, tmp_path / "second", formats="svg")
    assert (tmp_path / "first.svg").read_bytes() == (tmp_path / "second.svg").read_bytes()
    plt.close(fig)


def test_subfigure_result_exports_parent_canvas(tmp_path):
    fig = plt.figure()
    subfig = fig.subfigures(1, 2)[0]
    ax = subfig.subplots()
    result = PlotResult(subfig, {"main": ax})
    assert result.fig is fig
    assert result.metadata["export_scope"] == "parent_figure"
    assert "sibling panels" in result.metadata["source_scope"]
    assert result.save(tmp_path / "subfigure", formats="png")["figures"]["png"]
    plt.close(fig)


def test_style_context_restores_settings_even_on_exception():
    before = dict(matplotlib.rcParams)
    with pytest.raises(RuntimeError):
        with style_context("presentation", width_mm=100, overrides={"axes.labelsize": 18}):
            assert matplotlib.rcParams["axes.labelsize"] == 18
            assert matplotlib.rcParams["svg.fonttype"] == "none"
            raise RuntimeError("example failure")
    assert dict(matplotlib.rcParams) == before


def test_palette_is_stable_across_subsets_and_validates_overrides():
    full = category_palette(["T", "B", "Mono"])
    assert category_palette(["Mono", "T"])["T"] == full["T"]
    assert category_palette([1], {1: "red"}) == {"1": "red"}
    with pytest.raises(ValueError, match="does not contain"):
        category_palette(["T", "B"], {"T": "red"})


def test_display_selection_retains_rare_groups_and_required_cells():
    labels = np.array(["common"] * 99 + ["rare"], dtype=object)
    selected = display_indices(labels, 12, 4, required_indices=[2, 7])
    assert len(selected) == 12
    assert {2, 7, 99}.issubset(selected)
    np.testing.assert_array_equal(selected, display_indices(labels, 12, 4, required_indices=[2, 7]))
    with pytest.raises(ValueError, match="preserve"):
        display_indices(labels, 1)
    with pytest.raises(ValueError, match="required highlights"):
        display_indices(labels, 1, required_indices=[2, 7])


def test_embedding_aliases_have_identical_coordinates():
    from anndata import AnnData
    data = AnnData(np.zeros((3, 1)))
    data.obsm["X_umap"] = np.arange(6).reshape(3, 2)
    key, coords = resolve_embedding(data, "umap")
    assert key == "X_umap"
    np.testing.assert_array_equal(coords, resolve_embedding(data, "X_umap")[1])


def test_report_keeps_primary_effect_representation_for_table_plots():
    from anndata import AnnData

    import scgeo as sg
    data = AnnData(np.zeros((12, 1)), obs=pd.DataFrame({
        "condition": ["control"] * 6 + ["treated"] * 6,
        "sample": [f"sample_{i // 2}" for i in range(12)],
        "state": ["A"] * 12,
    }, index=[f"cell_{i}" for i in range(12)]))
    data.obsm["X_primary"] = np.arange(24, dtype=float).reshape(12, 2)
    sg.tl.robust_shift(data, rep="X_primary", condition_key="condition", group0="control",
                       group1="treated", sample_key="sample", by="state", n_boot=3)
    report = sg.get.state_report(data, node_key="state")
    assert report.attrs["provenance"]["effect_representation"] == "X_primary"
    plotted = sg.pl.state_evidence_panel(report, return_result=True, show=False)
    assert "X_primary" in plotted.axes["effect"].get_xlabel()
    plt.close(plotted.fig)
