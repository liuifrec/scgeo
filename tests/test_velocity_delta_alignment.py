import numpy as np
import pandas as pd
import anndata as ad


def test_velocity_delta_alignment_auto_key():
    rs = np.random.RandomState(0)
    n = 30
    d = 4

    obs = pd.DataFrame(
        {
            "condition": ["A"] * 15 + ["B"] * 15,
            "cell_type": ["T"] * 10 + ["B"] * 5 + ["T"] * 10 + ["B"] * 5,
        },
        index=[f"c{i}" for i in range(n)],
    )
    adata = ad.AnnData(X=np.zeros((n, 1)), obs=obs)

    # define UMAP embedding + velocity in same dim
    adata.obsm["X_umap"] = rs.normal(size=(n, d)).astype(np.float32)
    adata.obsm["velocity_umap"] = rs.normal(size=(n, d)).astype(np.float32)

    import scgeo as sg
    sg.tl.velocity_delta_alignment(
        adata,
        rep_for_shift="X_umap",
        condition_key="condition",
        group0="A",
        group1="B",
        obs_key="align",
    )

    assert "align" in adata.obs
    assert "shift" in adata.uns["scgeo"]
