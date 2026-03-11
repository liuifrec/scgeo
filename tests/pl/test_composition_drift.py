from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import matplotlib
matplotlib.use("Agg")

from scgeo.pl import composition_drift


def _toy_adata():
    # A decreases, B stable-ish, C increases
    obs = pd.DataFrame(
        {
            "node": pd.Categorical(
                ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
                categories=["A", "B", "C"],
            ),
            "timepoint": (
                ["D08"] * 5 + ["D21"] * 3 +   # A
                ["D08"] * 4 + ["D21"] * 4 +   # B
                ["D08"] * 2 + ["D21"] * 6     # C
            ),
        },
        index=[f"cell_{i}" for i in range(24)],
    )

    adata = AnnData(X=np.zeros((24, 3), dtype=float), obs=obs)
    adata.obsm["X_umap"] = np.array(
        [[0.0, 0.0]] * 8 +
        [[2.0, 0.0]] * 8 +
        [[4.0, 0.0]] * 8,
        dtype=float,
    )
    adata.uns["node_colors"] = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    return adata


def test_composition_drift_smoke_return_data():
    adata = _toy_adata()

    fig, axes, plot_df = composition_drift(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert len(axes) == 3
    assert isinstance(plot_df, pd.DataFrame)
    assert {"node", "n0", "n1", "frac0", "frac1", "delta_frac", "log2_fc"}.issubset(plot_df.columns)

    idx = plot_df.set_index("node")
    assert idx.loc["A", "delta_frac"] < 0
    assert idx.loc["C", "delta_frac"] > 0


def test_composition_drift_top_n_filters():
    adata = _toy_adata()

    _, _, plot_df = composition_drift(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        top_n=2,
        return_data=True,
        show=False,
    )

    assert len(plot_df) == 2


def test_composition_drift_sort_by_delta_frac():
    adata = _toy_adata()

    _, _, plot_df = composition_drift(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        sort_by="delta_frac",
        return_data=True,
        show=False,
    )

    vals = plot_df["delta_frac"].to_numpy()
    assert np.all(vals[:-1] >= vals[1:])


def test_composition_drift_missing_basis_raises():
    adata = _toy_adata()

    with pytest.raises(KeyError, match="Embedding"):
        composition_drift(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="tsne",
            show=False,
        )


def test_composition_drift_invalid_sort_by_raises():
    adata = _toy_adata()

    with pytest.raises(ValueError, match="sort_by"):
        composition_drift(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="umap",
            sort_by="weird_metric",
            show=False,
        )


def test_composition_drift_missing_obs_key_raises():
    adata = _toy_adata()

    with pytest.raises(KeyError, match="node"):
        composition_drift(
            adata,
            node_key="missing_node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="umap",
            show=False,
        )