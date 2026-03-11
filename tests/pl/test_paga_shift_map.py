from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import matplotlib
matplotlib.use("Agg")

from scipy import sparse

from scgeo.pl import paga_shift_map


def _toy_adata():
    # 12 cells, 3 nodes (A/B/C), 2 conditions (D08/D21)
    obs = pd.DataFrame(
        {
            "node": pd.Categorical(
                ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
                categories=["A", "B", "C"],
            ),
            "timepoint": ["D08", "D08", "D21", "D21"] * 3,
        },
        index=[f"cell_{i}" for i in range(12)],
    )

    X = np.zeros((12, 3), dtype=float)
    adata = AnnData(X=X, obs=obs)

    # UMAP-like layout:
    # A shifts right, B shifts up, C nearly stable
    adata.obsm["X_umap"] = np.array(
        [
            [0.0, 0.0], [0.2, 0.1], [1.0, 0.0], [1.2, 0.1],   # A
            [3.0, 0.0], [3.2, 0.1], [3.0, 1.0], [3.2, 1.1],   # B
            [6.0, 0.0], [6.2, 0.1], [6.1, 0.0], [6.3, 0.1],   # C
        ],
        dtype=float,
    )

    # simple chain A-B-C
    conn = np.array(
        [
            [0.0, 0.8, 0.0],
            [0.8, 0.0, 0.7],
            [0.0, 0.7, 0.0],
        ],
        dtype=float,
    )
    adata.uns["paga"] = {"connectivities": sparse.csr_matrix(conn)}
    adata.uns["node_colors"] = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    return adata


def test_paga_shift_map_smoke_return_data():
    adata = _toy_adata()

    fig, ax, cent, edges = paga_shift_map(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        paga_key="paga",
        min_cells=1,
        connectivity_threshold=0.05,
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert isinstance(cent, pd.DataFrame)
    assert {"node", "x0", "y0", "x1", "y1", "dx", "dy", "shift_umap"}.issubset(cent.columns)
    assert len(edges) == 2  # A-B and B-C

    cent_idx = cent.set_index("node")
    assert np.isclose(cent_idx.loc["A", "dx"], 1.0, atol=1e-6)
    assert np.isclose(cent_idx.loc["B", "dy"], 1.0, atol=1e-6)
    assert cent_idx.loc["A", "present0"]
    assert cent_idx.loc["A", "present1"]


def test_paga_shift_map_threshold_filters_edges():
    adata = _toy_adata()

    _, _, _, edges = paga_shift_map(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        connectivity_threshold=0.75,
        min_cells=1,
        return_data=True,
        show=False,
    )

    assert len(edges) == 1
    assert edges[0][0] == "A"
    assert edges[0][1] == "B"


def test_paga_shift_map_missing_basis_raises():
    adata = _toy_adata()
    with pytest.raises(KeyError, match="Embedding"):
        paga_shift_map(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="tsne",
            show=False,
        )


def test_paga_shift_map_min_cells_masks_arrows():
    adata = _toy_adata()

    # Make C sparse in D21 so it fails min_cells
    keep = ~((adata.obs["node"] == "C") & (adata.obs["timepoint"] == "D21") & (adata.obs_names == "cell_11"))
    adata = adata[keep].copy()

    fig, ax, cent, edges = paga_shift_map(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        min_cells=2,
        connectivity_threshold=0.05,
        return_data=True,
        show=False,
    )

    cent_idx = cent.set_index("node")
    assert bool(cent_idx.loc["C", "present0"]) is True
    assert bool(cent_idx.loc["C", "present1"]) is False
    assert fig is not None
    assert ax is not None


def test_paga_shift_map_accepts_existing_ax():
    import matplotlib.pyplot as plt

    adata = _toy_adata()
    fig, ax = plt.subplots(figsize=(4, 4))

    out = paga_shift_map(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        min_cells=1,
        ax=ax,
        show=False,
    )

    assert out is ax