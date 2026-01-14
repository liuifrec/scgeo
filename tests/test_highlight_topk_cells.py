import numpy as np
import pandas as pd
import anndata as ad
import scgeo as sg


def test_highlight_topk_cells_runs():
    n = 100
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_umap"] = np.random.RandomState(0).normal(size=(n, 2)).astype(np.float32)
    adata.obs["s"] = pd.Series(np.linspace(0, 1, n))
    fig, ax = sg.pl.highlight_topk_cells(adata, "s", topk=10, show=False)
    assert fig is not None and ax is not None
