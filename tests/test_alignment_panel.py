import numpy as np
import pandas as pd
import anndata as ad
import scgeo as sg


def test_alignment_panel_runs():
    n = 100
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_umap"] = np.random.RandomState(0).normal(size=(n, 2)).astype(np.float32)
    adata.obs["align"] = pd.Series(np.random.RandomState(1).rand(n), index=adata.obs.index)

    fig, axes = sg.pl.alignment_panel(adata, "align", show=False)
    assert fig is not None
    assert len(axes) == 2
