import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from scipy import sparse
from anndata import AnnData

import scgeo as sg


def _make_toy_adata():
    n = 24
    X = np.zeros((n, 5), dtype=float)
    adata = AnnData(X)

    # three clusters, two timepoints
    adata.obs["leiden_raw"] = pd.Categorical(
        ["0"] * 8 + ["1"] * 8 + ["2"] * 8,
        categories=["0", "1", "2"],
    )
    adata.obs["timepoint"] = pd.Categorical(
        ["D8"] * 4 + ["D21"] * 4 + ["D8"] * 4 + ["D21"] * 4 + ["D8"] * 4 + ["D21"] * 4,
        categories=["D8", "D21"],
    )

    # simple embedding
    xy = np.array(
        [[0, 0], [0.2, 0.1], [0.1, -0.2], [-0.1, 0.1],
         [0.5, 0.1], [0.6, 0.2], [0.5, -0.1], [0.4, 0.0],

         [2.0, 0.0], [2.1, 0.1], [2.0, -0.2], [1.9, 0.2],
         [2.5, 0.1], [2.6, 0.2], [2.4, -0.1], [2.5, 0.0],

         [4.0, 0.0], [4.1, 0.1], [4.2, -0.2], [3.9, 0.2],
         [4.5, 0.1], [4.6, 0.2], [4.4, -0.1], [4.5, 0.0]]
    )
    adata.obsm["X_umap"] = xy
    adata.obsm["velocity_umap"] = np.tile(np.array([[0.05, 0.02]]), (n, 1))

    # palette
    adata.uns["leiden_raw_colors"] = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    adata.uns["timepoint_colors"] = ["#4C78A8", "#E45756"]

    # paga connectivity
    conn = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.6, 0.2],
                [0.6, 0.0, 0.7],
                [0.2, 0.7, 0.0],
            ]
        )
    )
    adata.uns["paga"] = {"connectivities": conn}
    return adata


def test_paga_shift_map_basic():
    adata = _make_toy_adata()

    fig, ax, out = sg.pl.paga_shift_map(
        adata,
        node_key="leiden_raw",
        condition_key="timepoint",
        group0="D8",
        group1="D21",
        basis="umap",
        paga_key="paga",
        min_cells=2,
        show=False,
        return_data=True,
    )

    assert fig is not None
    assert ax is not None
    assert "centroids" in out
    assert "edges" in out
    assert len(out["edges"]) > 0
    assert "delta_frac" in out["centroids"].columns


def test_paga_scgeo_with_pies_and_velocity():
    adata = _make_toy_adata()

    fig, ax, out = sg.pl.paga_scgeo(
        adata,
        node_key="leiden_raw",
        condition_key="timepoint",
        group0="D8",
        group1="D21",
        basis="umap",
        paga_key="paga",
        pie_key="timepoint",
        velocity_basis="umap",
        show_velocity=True,
        node_color_mode="delta",
        min_cells=2,
        show=False,
        return_data=True,
    )

    assert fig is not None
    assert ax is not None
    assert "composition" in out
    assert "velocity" in out
    assert set(out["composition"].columns) >= {"node", "D8", "D21"}


def test_paga_shift_map_alignment_mode():
    adata = _make_toy_adata()
    align_df = pd.DataFrame(
        {
            "leiden_raw": ["0", "1", "2"],
            "alignment_cosine": [0.8, -0.4, 0.1],
        }
    )

    fig, ax = sg.pl.paga_shift_map(
        adata,
        node_key="leiden_raw",
        condition_key="timepoint",
        group0="D8",
        group1="D21",
        basis="umap",
        paga_key="paga",
        node_color_mode="alignment",
        alignment_df=align_df,
        min_cells=2,
        show=False,
    )

    assert fig is not None
    assert ax is not None