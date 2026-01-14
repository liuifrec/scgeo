import numpy as np
import pandas as pd
import anndata as ad
import scgeo as sg


def test_consensus_subspace_panel_runs():
    n = 80
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_umap"] = np.random.RandomState(0).normal(size=(n, 2)).astype(np.float32)
    adata.obs["cs_score"] = pd.Series(np.random.RandomState(1).rand(n), index=adata.obs.index)
    adata.uns["scgeo"] = {"consensus_subspace": {"singular_values": [3, 2, 1]}}

    fig, axes = sg.pl.consensus_subspace_panel(adata, show=False)
    assert fig is not None
    assert len(axes) == 2
