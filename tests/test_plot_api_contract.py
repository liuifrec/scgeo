"""Cross-API contracts and shared categorical rendering edge cases."""
import inspect
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import scgeo as sg


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_every_export_is_audited_and_matplotlib_renderers_are_composable():
    utilities = {"PlotResult", "export_figure", "category_palette", "style_context",
                 "_get_scanpy_palette", "_fallback_palette"}
    native = {"state_flow_sankey", "perturbation_report"}
    aliases = {"score_umap": "score_embedding", "paga_scgeo": "paga_shift_map"}
    audit = (Path(__file__).parents[1] / "docs" / "PLOTTING_API_STATUS.md").read_text()
    exported = sg.pl.__all__
    assert len(exported) == len(set(exported))
    for name in exported:
        assert f"| `{name}` |" in audit, f"Unaudited plotting export: {name}"
        if name in utilities | native:
            continue
        renderer = getattr(sg.pl, aliases.get(name, name))
        params = inspect.signature(renderer).parameters
        assert "return_result" in params, name
        # The atlas shipped in PR #20 with PlotResult as its default.
        assert params["return_result"].default is (name == "geometry_dynamics_atlas"), name
        assert {"ax", "axes"} & set(params), name


def fixture(labels):
    adata = AnnData(obs=pd.DataFrame({"label": labels}, index=["duplicate"] * len(labels)))
    adata.obsm["X_display"] = np.column_stack((np.arange(len(labels), dtype=float), np.zeros(len(labels))))
    return adata


def test_score_categories_do_not_overwrite_auxiliary_artists_or_missingness():
    labels = ["Missing", "missing", "legend", "categories", "(missing)", None]
    with pytest.warns(UserWarning, match="not unique"):
        adata = fixture(labels)
    result = sg.pl.score_embedding(adata, "label", basis="display", return_result=True, show=False)
    assert result.metadata["missing_label"] == "((missing))"
    assert result.data.value.isna().sum() == 1
    assert result.data.value_label.tolist() == labels[:-1] + ["((missing))"]
    assert result.data.obs_position.tolist() == list(range(6))
    assert len(result.artists["missing"].get_offsets()) == 1
    assert set(result.artists["categories"]) == set(labels[:-1])
    assert all(isinstance(artist, PathCollection) for artist in result.artists["categories"].values())
    assert len(result.artists["legend"].get_texts()) == 6
    assert result.metadata["export_scope"] == "created_figure"


@pytest.mark.parametrize("labels,kwargs,match", [
    ([1, "1"], {}, "colliding"),
    (["A", "B"], {"vmin": 0}, "categorical"),
    (["A", "B"], {"palette": {"A": "red"}}, "palette"),
])
def test_invalid_category_inputs_fail_before_creating_figures(labels, kwargs, match):
    with pytest.warns(UserWarning, match="not unique"):
        adata = fixture(labels)
    before = plt.get_fignums()
    with pytest.raises(ValueError, match=match):
        sg.pl.score_embedding(adata, "label", basis="display", show=False, **kwargs)
    assert plt.get_fignums() == before


def test_declared_unused_missing_label_is_reserved():
    with pytest.warns(UserWarning, match="not unique"):
        adata = fixture(pd.Categorical(["A", None], categories=["A", "(missing)"]))
    result = sg.pl.score_embedding(adata, "label", basis="display", return_result=True, show=False)
    assert result.metadata["missing_label"] == "((missing))"
    assert result.data.value_label.tolist() == ["A", "((missing))"]
