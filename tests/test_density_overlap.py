import numpy as np
import pandas as pd
import anndata as ad


def test_density_overlap_runs_and_bounds():
    rs = np.random.RandomState(0)
    n = 200
    d = 2
    # Two slightly shifted clouds
    X0 = rs.normal(0, 1, size=(n, d)).astype(np.float32)
    X1 = rs.normal(0.6, 1, size=(n, d)).astype(np.float32)
    X = np.vstack([X0, X1])

    obs = pd.DataFrame(
        {
            "condition": ["A"] * n + ["B"] * n,
            "cell_type": (["T"] * (n // 2) + ["B"] * (n // 2)) * 2,
        },
        index=[f"c{i}" for i in range(2 * n)],
    )
    adata = ad.AnnData(X=np.zeros((2 * n, 1)), obs=obs)
    adata.obsm["X_umap"] = X

    import scgeo as sg
    sg.tl.density_overlap(
        adata,
        rep="X_umap",
        condition_key="condition",
        group0="A",
        group1="B",
        by="cell_type",
        k=15,
    )

    out = adata.uns["scgeo"]["density_overlap"]
    bc = out["global"]["bc"]
    hell = out["global"]["hellinger"]

    assert 0.0 <= bc <= 1.0
    assert 0.0 <= hell <= 1.0
    assert "by" in out
