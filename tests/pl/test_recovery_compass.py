from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import matplotlib
matplotlib.use("Agg")

from scipy import sparse

from scgeo.pl import recovery_compass


def _toy_adata():
    obs = pd.DataFrame(
        {
            "node": pd.Categorical(
                ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
                categories=["A", "B", "C"],
            ),
            "timepoint": ["D08", "D08", "D21", "D21"] * 3,
            "scgeo_ood": [0.10, 0.12, 0.14, 0.18, 0.20, 0.22, 0.60, 0.70, 0.30, 0.35, 0.80, 0.95],
        },
        index=[f"cell_{i}" for i in range(12)],
    )

    adata = AnnData(X=np.zeros((12, 3), dtype=float), obs=obs)

    # A shifts right, B shifts up, C shifts left/down
    adata.obsm["X_umap"] = np.array(
        [
            [0.0, 0.0], [0.2, 0.0], [1.0, 0.0], [1.2, 0.0],   # A
            [3.0, 0.0], [3.2, 0.0], [3.0, 1.0], [3.2, 1.0],   # B
            [6.0, 1.0], [6.2, 1.0], [5.7, 0.7], [5.9, 0.7],   # C
        ],
        dtype=float,
    )

    adata.obsm["velocity_umap"] = np.array(
        [
            [0.8, 0.0], [1.0, 0.0], [0.9, 0.0], [1.1, 0.0],   # A aligned
            [0.0, 0.8], [0.0, 1.0], [0.0, 0.9], [0.0, 1.1],   # B aligned
            [0.6, 0.6], [0.5, 0.5], [0.6, 0.5], [0.5, 0.6],   # C anti-ish
        ],
        dtype=float,
    )

    conn = np.array(
        [
            [0.0, 0.8, 0.0],
            [0.8, 0.0, 0.7],
            [0.0, 0.7, 0.0],
        ],
        dtype=float,
    )
    adata.uns["paga"] = {"connectivities": sparse.csr_matrix(conn)}
    return adata


def test_recovery_compass_smoke_return_data():
    adata = _toy_adata()

    fig, ax, node_df, edges = recovery_compass(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        ood_key="scgeo_ood",
        min_cells=1,
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert isinstance(node_df, pd.DataFrame)
    assert len(edges) == 2
    assert {"node", "dx", "dy", "shift_norm", "alignment_cosine", "ood_frac", "delta_frac"}.issubset(node_df.columns)

    idx = node_df.set_index("node")
    assert np.isclose(idx.loc["A", "dx"], 1.0, atol=1e-6)
    assert np.isclose(idx.loc["B", "dy"], 1.0, atol=1e-6)
    assert idx.loc["A", "alignment_cosine"] > 0.9
    assert idx.loc["B", "alignment_cosine"] > 0.9


def test_recovery_compass_missing_basis_raises():
    adata = _toy_adata()
    with pytest.raises(KeyError, match="Embedding"):
        recovery_compass(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="tsne",
            show=False,
        )


def test_recovery_compass_missing_paga_raises():
    adata = _toy_adata()
    del adata.uns["paga"]

    with pytest.raises(KeyError, match="paga"):
        recovery_compass(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="umap",
            show=False,
        )


def test_recovery_compass_missing_velocity_is_tolerated():
    adata = _toy_adata()
    del adata.obsm["velocity_umap"]

    fig, ax, node_df, edges = recovery_compass(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        ood_key="scgeo_ood",
        min_cells=1,
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert "alignment_cosine" in node_df.columns
    assert node_df["alignment_cosine"].isna().all()


def test_recovery_compass_missing_ood_is_tolerated():
    adata = _toy_adata()
    del adata.obs["scgeo_ood"]

    fig, ax, node_df, edges = recovery_compass(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        ood_key=None,
        min_cells=1,
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert "ood_frac" in node_df.columns
    assert node_df["ood_frac"].isna().all()


def test_recovery_compass_invalid_modes_raise():
    adata = _toy_adata()

    with pytest.raises(ValueError, match="node_size_mode"):
        recovery_compass(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="umap",
            node_size_mode="weird",
            show=False,
        )

    with pytest.raises(ValueError, match="fill_color_mode"):
        recovery_compass(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="umap",
            fill_color_mode="weird",
            show=False,
        )

    with pytest.raises(ValueError, match="arrow_color_mode"):
        recovery_compass(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="umap",
            arrow_color_mode="weird",
            show=False,
        )


def test_recovery_compass_accepts_existing_ax():
    import matplotlib.pyplot as plt

    adata = _toy_adata()
    fig, ax = plt.subplots(figsize=(5, 5))

    out = recovery_compass(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        ood_key="scgeo_ood",
        min_cells=1,
        ax=ax,
        show=False,
    )

    assert out is ax