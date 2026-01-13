import numpy as np
import pandas as pd
import anndata as ad


def test_projection_disagreement_basic():
    rs = np.random.RandomState(0)
    n, d = 25, 3
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=np.zeros((n, 1)), obs=obs)

    adata.obsm["v1"] = rs.normal(size=(n, d)).astype(np.float32)
    adata.obsm["v2"] = adata.obsm["v1"].copy()

    import scgeo as sg
    sg.tl.projection_disagreement(
        adata,
        sources=[
            {"type": "obsm", "key": "v1", "name": "v1"},
            {"type": "obsm", "key": "v2", "name": "v2"},
        ],
        obs_key="dis",
    )

    assert "dis" in adata.obs
    # identical vectors => cosine ~1 => disagreement ~0
    assert float(np.nanmax(adata.obs["dis"].values)) < 0.02
