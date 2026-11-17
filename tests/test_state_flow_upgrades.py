"""State-association plots conserve observed cells, including omitted ribbons."""

import hashlib
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


def adata_from_obs(values):
    obs = pd.DataFrame(values)
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    return AnnData(np.zeros((len(obs), 1)), obs=obs)


@pytest.fixture
def flow_data():
    pairs = [("A", "X")] * 12 + [("A", "Y")] * 6
    pairs += [("B", "X")] * 4 + [("B", "Y")] * 2
    data = adata_from_obs({"left": [p[0] for p in pairs],
                          "right": [p[1] for p in pairs]})
    data.obs["left"] = pd.Categorical(data.obs.left, categories=["B", "A"])
    data.obs["right"] = pd.Categorical(data.obs.right, categories=["Y", "X"])
    return data


def plot(kind, data, **kwargs):
    options = dict(columns=["left", "right"], return_result=True, show=False)
    options.update(kwargs)
    return getattr(sg.pl, "state_flow_" + kind)(data, **options)


def snapshot(ax):
    return (list(ax.get_children()), ax.get_xlim(), ax.get_ylim(),
            ax.get_title(), ax.figure.get_size_inches().copy())


def assert_unchanged(ax, before):
    assert list(ax.get_children()) == before[0]
    assert ax.get_xlim() == before[1]
    assert ax.get_ylim() == before[2]
    assert ax.get_title() == before[3]
    np.testing.assert_array_equal(ax.figure.get_size_inches(), before[4])


@pytest.mark.parametrize("kind", ["alluvial", "chord"])
def test_counts_and_filtered_denominators_reconcile(kind, flow_data):
    result = plot(kind, flow_data, min_count=5)
    links = result.data["links"]
    nodes = result.data["nodes"]
    expected = {("A", "X"): 12, ("A", "Y"): 6, ("B", "X"): 4, ("B", "Y"): 2}
    assert len(links) == 4
    for row in links.itertuples():
        assert row.count == expected[row.source_label, row.target_label]
        assert row.fraction == pytest.approx(row.count / 24)
        assert row.displayed == (row.count >= 5)
    assert links[links.displayed]["count"].sum() == 18
    assert links[~links.displayed]["count"].sum() == 6
    assert links[links.displayed].fraction.sum() == pytest.approx(.75)
    assert nodes.groupby("stage")["count"].sum().to_dict() == {"left": 24, "right": 24}
    assert nodes.groupby("stage").fraction.sum().tolist() == pytest.approx([1, 1])
    assert result.data["cells"].included.sum() == 24
    assert result.data["cells"].cell.tolist() == flow_data.obs_names.tolist()


@pytest.mark.parametrize("kind", ["alluvial", "chord"])
def test_normalize_does_not_inflate_filtered_links(kind, flow_data):
    count_labels = plot(kind, flow_data, min_count=5, normalize=False)
    percent_labels = plot(kind, flow_data, min_count=5, normalize=True)
    for key in ["nodes", "links"]:
        pd.testing.assert_frame_equal(count_labels.data[key], percent_labels.data[key])
    assert percent_labels.data["links"].query("displayed").fraction.sum() == pytest.approx(.75)


@pytest.mark.parametrize("kind", ["alluvial", "chord"])
def test_allocated_slots_keep_full_marginals_and_omitted_gaps(kind, flow_data):
    result = plot(kind, flow_data, min_count=5)
    nodes = result.data["node_layout"].set_index("node_id")
    links = result.data["link_layout"]
    assert len(links) == 4
    assert not links.query("count < 5").displayed.any()
    for side in ["source", "target"]:
        for node_id, subset in links.groupby(side + "_id"):
            node = nodes.loc[node_id]
            if kind == "alluvial":
                prefix = "y0" if side == "source" else "y1"
                low, high = subset[prefix + "_low"], subset[prefix + "_high"]
                span = node.y1 - node.y0
                assert low.min() >= node.y0 - 1e-12
                assert high.max() <= node.y1 + 1e-12
                widths = high - low
            else:
                start, end = subset[side + "_theta_start"], subset[side + "_theta_end"]
                span = node.theta_high - node.theta_low
                assert np.minimum(start, end).min() >= node.theta_low - 1e-10
                assert np.maximum(start, end).max() <= node.theta_high + 1e-10
                widths = np.abs(end - start)
            assert widths.sum() == pytest.approx(span)
            np.testing.assert_allclose(widths / span, subset["count"] / node["count"])
            assert widths[subset.displayed].sum() <= span + 1e-10


def test_three_stage_links_use_same_population_and_pairwise_counts(flow_data):
    flow_data.obs["third"] = ["one"] * 10 + ["two"] * 14
    result = plot("alluvial", flow_data, columns=["left", "right", "third"], min_count=5)
    links = result.data["links"]
    assert links.groupby("pair_index")["count"].sum().tolist() == [24, 24]
    assert result.data["nodes"].groupby("stage")["count"].sum().tolist() == [24, 24, 24]
    for pair_index, (left, right) in enumerate(zip(["left", "right"], ["right", "third"])):
        expected = flow_data.obs.groupby([left, right], observed=True).size()
        actual = links[links.pair_index == pair_index]
        for row in actual.itertuples():
            assert row.count == expected.loc[(row.source_label, row.target_label)]


@pytest.mark.parametrize("kind", ["alluvial", "chord"])
def test_missing_values_are_distinct_from_literal_na_and_missing_labels(kind):
    data = adata_from_obs({"left": [None, "NA", "nan", "(missing)", pd.NA, "ok"],
                          "right": ["X"] * 6})
    complete = plot(kind, data)
    left = complete.data["nodes"].query("stage == 'left'")
    assert left[left.is_missing]["count"].sum() == 2
    assert left.loc[left.label.eq("NA"), "count"].sum() == 1
    assert left.loc[left.label.eq("nan"), "count"].sum() == 1
    assert left.loc[left.label.eq("(missing)"), "count"].sum() == 1
    assert left.display_label.is_unique
    dropped = plot(kind, data, drop_na=True)
    assert len(dropped.data["cells"]) == 6
    assert dropped.data["cells"].included.sum() == 4
    assert dropped.data["links"]["count"].sum() == 4
    assert "NA" in dropped.data["nodes"].label.tolist()


@pytest.mark.parametrize("kind", ["alluvial", "chord"])
def test_empty_string_category_renders_as_nonmissing_label(kind):
    data = adata_from_obs({"left": ["", "A", None],
                          "right": ["X", "X", "X"]})
    result = plot(kind, data, label_format="label")
    result.fig.canvas.draw()
    nodes = result.data["nodes"].query("stage == 'left'")
    blank = nodes[nodes.label.eq("")]
    assert len(blank) == 1
    assert not blank.is_missing.iloc[0]
    assert blank["count"].iloc[0] == 1
    assert nodes[nodes.is_missing]["count"].sum() == 1
    assert result.data["links"]["count"].sum() == 3


@pytest.mark.parametrize("kind", ["alluvial", "chord"])
def test_stage_identity_does_not_collide_on_repeated_labels_or_delimiters(kind):
    data = adata_from_obs({"a::b": ["c", "same", "same"],
                          "a": ["b::c", "same", "same"]})
    result = plot(kind, data, columns=["a::b", "a"])
    nodes = result.data["nodes"]
    assert len(nodes) == 4
    assert nodes.node_id.is_unique
    assert nodes.node_key.is_unique
    assert len(nodes[nodes.label.eq("same")]) == 2
    assert result.data["links"]["count"].sum() == 3
    assert (result.data["links"].source_id != result.data["links"].target_id).all()


@pytest.mark.parametrize("kind", ["alluvial", "chord"])
def test_category_order_changes_layout_without_changing_counts(kind, flow_data):
    original = plot(kind, flow_data)
    assert original.data["nodes"].query("stage == 'left'").label.tolist() == ["B", "A"]
    assert original.data["nodes"].query("stage == 'right'").label.tolist() == ["Y", "X"]
    custom = plot(kind, flow_data, category_order={"left": ["A", "B"], "right": ["X", "Y"]})
    assert custom.data["nodes"].query("stage == 'left'").label.tolist() == ["A", "B"]
    keys = ["source_stage", "target_stage", "source_label", "target_label"]
    pd.testing.assert_series_equal(
        original.data["links"].set_index(keys)["count"].sort_index(),
        custom.data["links"].set_index(keys)["count"].sort_index(),
    )
    reordered = plot(kind, flow_data, order="barycentric")
    pd.testing.assert_series_equal(
        original.data["links"].set_index(keys)["count"].sort_index(),
        reordered.data["links"].set_index(keys)["count"].sort_index(),
    )


@pytest.mark.parametrize("kind", ["alluvial", "chord"])
def test_all_filtered_preserves_nodes_and_explicit_omissions(kind, flow_data):
    result = plot(kind, flow_data, min_count=100)
    assert result.data["nodes"]["count"].sum() == 48
    assert result.data["links"]["count"].sum() == 24
    assert not result.data["links"].displayed.any()
    assert not result.data["link_layout"].displayed.any()
    assert result.warnings


def test_many_alluvial_categories_are_not_clipped():
    labels = [f"Category {i:02d}" for i in range(16)]
    data = adata_from_obs({"left": labels, "right": labels[::-1]})
    result = plot("alluvial", data, category_gap=.02, figsize=(12, 8))
    ax = result.axes["alluvial"]
    nodes = result.data["node_layout"]
    assert nodes.y0.min() >= min(ax.get_ylim())
    assert nodes.y1.max() <= max(ax.get_ylim())
    assert ((nodes.y1 - nodes.y0) > 0).all()


@pytest.mark.parametrize("kind", ["alluvial", "chord"])
def test_legacy_returns_and_supplied_axes_are_preserved(kind, flow_data):
    fn = getattr(sg.pl, "state_flow_" + kind)
    fig, ax = fn(flow_data, columns=["left", "right"], show=False)
    size = fig.get_size_inches().copy()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    assert fn(flow_data, columns=["left", "right"], ax=ax, show=False) is ax
    assert ax.get_xlim() == (-5, 5)
    assert ax.get_ylim() == (-4, 4)
    np.testing.assert_array_equal(fig.get_size_inches(), size)
    returned = fn(flow_data, columns=["left", "right"], return_data=True, show=False)
    assert len(returned) == 4
    assert not returned[2].empty and not returned[3].empty
    if kind == "alluvial":
        assert {"stage", "label", "x", "y0", "y1", "value"} <= set(returned[2])
        assert {"left_stage", "right_stage", "left_label", "right_label", "value"} <= set(returned[3])


@pytest.mark.parametrize("kind", ["alluvial", "chord"])
def test_subfigure_export_preserves_source_counts_and_hashes(kind, flow_data, tmp_path):
    parent = plt.figure(figsize=(10, 5))
    subfig = parent.subfigures(1, 2)[0]
    ax = subfig.subplots()
    result = plot(kind, flow_data, ax=ax, min_count=5)
    assert result.fig is parent
    assert result.axes[kind] is ax
    assert result.metadata["export_scope"] == "parent_figure"
    assert "sibling" in result.metadata["source_scope"]
    exported = result.save(tmp_path / kind, formats="svg")
    manifest = json.loads(Path(exported["metadata"]).read_text())
    for key in ["cells", "nodes", "links", "node_layout", "link_layout", "labels"]:
        path = Path(exported["tables"][key])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["tables"][key]["sha256"]
    links = pd.read_csv(exported["tables"]["links"])
    assert links["count"].sum() == 24
    assert links[links.displayed].fraction.sum() == pytest.approx(.75)


@pytest.mark.parametrize("kind", ["alluvial", "chord"])
@pytest.mark.parametrize("options", [
    {"columns": ["left"]}, {"columns": ["left", "left"]},
    {"min_count": 0}, {"min_count": 1.5}, {"alpha": -1},
    {"color_by": "unknown"}, {"order": "unknown"},
    {"category_order": {"left": ["A"]}},
    {"palette": {"A": "red"}},
])
def test_invalid_options_fail_before_axes_mutation(kind, flow_data, options):
    _, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    ax.set(title="Caller", xlim=(-3, 3), ylim=(-3, 3))
    before = snapshot(ax)
    with pytest.raises((ValueError, TypeError, KeyError)):
        plot(kind, flow_data, ax=ax, **options)
    assert_unchanged(ax, before)


@pytest.mark.parametrize("kind", ["alluvial", "chord"])
def test_string_conversion_collisions_are_rejected_before_drawing(kind):
    data = adata_from_obs({"left": [1, "1"], "right": ["X", "Y"]})
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="(?i)(label|distinct|string)"):
        plot(kind, data)
    assert plt.get_fignums() == before


def test_chord_requires_exactly_two_stages(flow_data):
    flow_data.obs["third"] = "Z"
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="(?i)(two|2)"):
        plot("chord", flow_data, columns=["left", "right", "third"])
    assert plt.get_fignums() == before


def test_sankey_uses_same_counts_and_literal_na_policy():
    pytest.importorskip("plotly")
    data = adata_from_obs({"left": [None, "NA", "NA", "A", "A", "A"],
                          "right": ["X", "Y", "Y", "X", "X", "Y"]})
    fig, nodes, links = sg.pl.state_flow_sankey(
        data, columns=["left", "right"], min_count=2,
        drop_na=True, return_data=True, show=False)
    assert "NA" in nodes.label.tolist()
    assert links.value.sum() == 4
    assert dict(zip(zip(links.source_label, links.target_label), links.value)) == {
        ("NA", "Y"): 2, ("A", "X"): 2,
    }
    assert list(fig.data[0].link.value) == links.value.tolist()
    assert fig.layout.meta is not None


def test_input_observations_are_not_modified(flow_data):
    obs = flow_data.obs.copy(deep=True)
    plot("alluvial", flow_data, order="barycentric")
    plot("chord", flow_data, normalize=True)
    pd.testing.assert_frame_equal(flow_data.obs, obs)
