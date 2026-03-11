from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

import matplotlib
matplotlib.use("Agg")

from scipy import sparse

from scgeo.pl import gallery_overview


def _toy_adata():
    obs = pd.DataFrame(
        {
            "node": pd.Categorical(
                ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
                categories=["A", "B", "C"],
            ),
            "timepoint": ["D08", "D08", "D21", "D21"] * 3,
            "scgeo_ood": [0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.6, 0.7, 0.3, 0.4, 0.8, 0.9],
        },
        index=[f"cell_{i}" for i in range(12)],
    )
    adata = AnnData(X=np.zeros((12, 2)), obs=obs)
    adata.obsm["X_umap"] = np.array(
        [
            [0.0, 0.0], [0.2, 0.0], [1.0, 0.0], [1.2, 0.0],
            [3.0, 0.0], [3.2, 0.0], [3.0, 1.0], [3.2, 1.0],
            [6.0, 1.0], [6.2, 1.0], [5.7, 0.7], [5.9, 0.7],
        ],
        dtype=float,
    )
    adata.obsm["velocity_umap"] = np.array(
        [
            [0.8, 0.0], [1.0, 0.0], [0.9, 0.0], [1.1, 0.0],
            [0.0, 0.8], [0.0, 1.0], [0.0, 0.9], [0.0, 1.1],
            [0.6, 0.6], [0.5, 0.5], [0.6, 0.5], [0.5, 0.6],
        ],
        dtype=float,
    )
    conn = np.array(
        [
            [0.0, 0.8, 0.0],
            [0.8, 0.0, 0.7],
            [0.0, 0.7, 0.0],
        ]
    )
    adata.uns["paga"] = {"connectivities": sparse.csr_matrix(conn)}
    adata.uns["node_colors"] = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    return adata


def test_gallery_overview_smoke():
    adata = _toy_adata()
    fig, axes = gallery_overview(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        ood_key="scgeo_ood",
        show=False,
    )
    assert fig is not None
    assert len(axes) == 4