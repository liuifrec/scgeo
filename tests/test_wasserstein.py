import numpy as np
import pandas as pd
import anndata as ad


def test_wasserstein_basic_positive():
    rs = np.random.RandomState(0)
    n = 100
    d = 5

    # Two Gaussians with a mean shift
    X0 = rs.normal(loc=0.0, scale=1.0, size=(n, d)).astype(np.float32)
    X1 = rs.normal(loc=1.0, scale=1.0, size=(n, d)).astype(np.float32)
    X = np.vstack([X0, X1])

    obs = pd.DataFrame(
        {
            "condition": ["A"] * n + ["B"] * n,
            "cell_type": (["T"] * (n // 2) + ["B"] * (n // 2)) * 2,
        },
        index=[f"c{i}" for i in range(2 * n)],
    )
    adata = ad.AnnData(X=np.zeros((2 * n, 1)), obs=obs)
    adata.obsm["X_pca"] = X

    import scgeo as sg
    sg.tl.wasserstein(
        adata,
        rep="X_pca",
        condition_key="condition",
        group0="A",
        group1="B",
        n_proj=64,
        seed=0,
    )

    swd = adata.uns["scgeo"]["wasserstein"]["global"]["swd"]
    assert swd > 0.2  # should be noticeably > 0 for shifted Gaussians


def test_wasserstein_by_levels_runs():
    rs = np.random.RandomState(1)
    n = 60
    d = 3
    X = rs.normal(size=(2 * n, d)).astype(np.float32)

    obs = pd.DataFrame(
        {
            "condition": ["A"] * n + ["B"] * n,
            "cell_type": (["T"] * 30 + ["B"] * 30) * 2,
        },
        index=[f"c{i}" for i in range(2 * n)],
    )
    adata = ad.AnnData(X=np.zeros((2 * n, 1)), obs=obs)
    adata.obsm["X_pca"] = X

    import scgeo as sg
    sg.tl.wasserstein(
        adata,
        rep="X_pca",
        condition_key="condition",
        group0="A",
        group1="B",
        by="cell_type",
        n_proj=16,
        seed=0,
    )

    assert "by" in adata.uns["scgeo"]["wasserstein"]
    assert set(adata.uns["scgeo"]["wasserstein"]["by"].keys()) >= {"T", "B"}
