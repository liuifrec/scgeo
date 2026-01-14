import numpy as np
import pandas as pd
import anndata as ad
import scgeo as sg


def test_embedding_density_runs():
    n = 200
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_umap"] = np.random.RandomState(0).normal(size=(n, 2)).astype(np.float32)
    adata.obs["grp"] = pd.Series(
    ["A"] * 80 + ["B"] * 70 + ["C"] * 50,
    index=adata.obs.index,
    )


    fig, axes = sg.pl.embedding_density(adata, "grp", show=False)
    assert fig is not None
    assert len(axes) == 3
