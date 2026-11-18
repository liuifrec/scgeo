"""Legacy summary contracts and auditable display-space recovery encodings."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
    adata = AnnData(obs=pd.DataFrame({
        "state": pd.Categorical(["A"]*4+["B"]*4, categories=["B", "A", "C"]),
        "condition": ["control", "control", "treated", "treated"]*2,
        "score": pd.array([0., .1, .2, .3, .4, .5, None, 1.], dtype="Float64"),
        "ood": np.arange(8.)/8,
    }, index=[f"c{i}" for i in range(8)]))
    adata.obsm["X_umap"] = np.array([[0,0], [.2,0], [1,0], [1.2,0], [3,0], [3.2,0], [3,2], [3.2,2]])
    adata.obsm["velocity_umap"] = np.array([[1.,0]]*4+[[0.,1]]*4)
    adata.uns["state_colors"] = ["#357B8A", "#B55E3C", "#ACACAC"]
    adata.uns["paga"] = {"groups": "state", "node_order": ["A", "B", "C"],
        "connectivities": np.array([[0.,.8,0.],[.8,0.,.2],[0.,.2,0.]])}
    return adata


def test_structure_selection_source_and_shared_norm_preserve_parent(data, tmp_path):
    data.obsm["X_umap"][7] = np.nan
    fig = plt.figure(figsize=(9, 4))
    subfigs = fig.subfigures(1, 2)
    axs = subfigs[0].subplots(1, 2)
    sibling = subfigs[1].subplots()
    bounds = sibling.get_position().bounds
    norm = Normalize(0, 1)
    result = sg.pl.consensus_structure(data, score_key="score", groupby="state", basis="X_umap",
        topk=3, min_frac_in_legend=.8, ax=axs[0], cax=axs[1], norm=norm,
        print_legend=False, return_result=True, show=False)
    assert result.fig is fig and result.metadata["export_scope"] == "parent_figure"
    assert sibling.get_position().bounds == bounds
    assert result.data["cells"].selected.sum() == 3
    assert result.data["cells"].highlighted.sum() == 2
    assert result.data["summary"]["count"].sum() == 3
    assert result.data["summary"].set_index("group").loc["B", "count"] == 3
    assert result.artists["selected"].norm is norm
    assert result.artists["colorbar"].ax is axs[1]
    output = result.save(tmp_path / "structure", formats="svg")
    assert len(pd.read_csv(output["tables"]["cells"])) == data.n_obs


def test_structure_missing_labels_and_legacy_return(data):
    data.obs["labels"] = [None, "nan", "A", "A", "A", "B", "B", "B"]
    legacy = sg.pl.consensus_structure(data, score_key="score", groupby="labels",
        topk=7, print_legend=False, colorbar=False, show=False)
    assert len(legacy) == 4
    assert legacy[2].group_counts["(missing)"] == 1
    assert legacy[2].group_counts["nan"] == 1
    empty = sg.pl.consensus_structure(data, score_key="score", groupby="labels", topk=0,
        print_legend=False, return_result=True, show=False)
    assert not empty.data["cells"].selected.any()
    assert empty.data["summary"]["count"].sum() == 0


def test_legend_truncation_preserves_full_entries_and_caller_axes(tmp_path):
    entries = [{"group": "A", "count": 3, "fraction": .5, "color": "red"},
               {"group": "B", "fraction": .5, "color": "blue"}, {"group": "C"}]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Parent title")
    bounds = [ax.get_position().bounds for ax in axs]
    result = sg.pl.legend_from_data(iter(entries), ax=axs[0], max_items=2, return_result=True, show=False)
    assert len(result.data) == 3 and result.data.displayed.tolist() == [True, True, False]
    assert [t.get_text() for t in result.artists["legend"].get_texts()] == ["A (50.0%, n=3)", "B (50.0%, n=0)"]
    assert result.warnings
    assert fig._suptitle.get_text() == "Parent title"
    np.testing.assert_equal([ax.get_position().bounds for ax in axs], bounds)
    output = result.save(tmp_path / "legend", formats="svg")
    assert len(pd.read_csv(output["tables"]["data"])) == 3
    assert len(sg.pl.legend_from_data([], show=False)) == 2


def compass(adata, **kwargs):
    return sg.pl.recovery_compass(adata, node_key="state", condition_key="condition",
        group0="control", group1="treated", basis="X_umap", min_cells=2, show=False, **kwargs)


def test_compass_graph_order_sources_and_exact_arrow_endpoints(data, tmp_path):
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    bounds = [ax.get_position().bounds for ax in axs]
    obs_before = data.obs.copy(deep=True)
    result = compass(data, ax=axs[0], caxes={"fill": axs[1], "arrow": axs[2]},
        arrow_scale=2., ood_key="ood", connectivity_threshold=.5, return_result=True)
    assert result.fig is fig and len(fig.axes) == 4
    assert result.metadata["graph_node_order"] == ["A", "B", "C"]
    assert result.metadata["graph_order_source"] == "stored_node_order"
    assert result.metadata["alignment_source"] == "display_coordinates"
    assert result.metadata["velocity_scope"] == "all_input_conditions"
    assert len(result.data["edges"]) == 2 and result.data["edges"].displayed.sum() == 1
    glyph = result.data["glyphs"].set_index("node")
    np.testing.assert_allclose(glyph.loc["A", ["x", "y", "arrow_end_x", "arrow_end_y"]].to_numpy(float), [.1, 0., 2.1, 0.])
    np.testing.assert_allclose(glyph.loc["B", ["x", "y", "arrow_end_x", "arrow_end_y"]].to_numpy(float), [3.1, 0., 3.1, 4.])
    np.testing.assert_allclose([ax.get_position().bounds for ax in axs], bounds)
    pd.testing.assert_frame_equal(data.obs, obs_before)
    output = result.save(tmp_path / "compass", formats="svg")
    meta = json.loads(Path(output["metadata"]).read_text())
    assert len(meta["tables"]) == 5
    assert len(pd.read_csv(output["tables"]["cells"])) == 8
    assert meta["plot"]["ood_threshold"] == np.quantile(data.obs.ood, .95)
    _, _, nodes, edges = compass(data, return_data=True)
    assert len(nodes) == 2 and edges == [("A", "B", .8)]


@pytest.mark.parametrize("bad_graph,match", [
    ({"groups": "wrong"}, "groups metadata"),
    ({"connectivities": np.zeros((2,2))}, "shape"),
    ({"connectivities": np.array([[0.,.1,0.],[.8,0.,.2],[0.,.2,0.]])}, "symmetric"),
])
def test_compass_rejects_mismatched_graph_without_drawing(data, bad_graph, match):
    data.uns["paga"].update(bad_graph)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=match):
        compass(data, ax=ax)
    assert not ax.collections and not ax.lines


def test_compass_missing_velocity_and_insufficient_condition_stay_unavailable(data):
    data.obs.loc["c7", "condition"] = "other"
    result = compass(data, legend=False, return_result=True)
    glyphs = result.data["glyphs"].set_index("node")
    assert not glyphs.loc["B", "arrow_displayed"]
    assert not glyphs.loc["B", "fill_available"]
    del data.obsm["velocity_umap"]
    missing = compass(data, return_result=True)
    assert missing.data["nodes"].alignment_cosine.isna().all()
    assert not missing.data["glyphs"].fill_available.any()
    assert any("alignment remains missing" in w for w in missing.warnings)
