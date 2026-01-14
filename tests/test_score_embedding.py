import numpy as np
import pandas as pd
import anndata as ad

import scgeo as sg


def test_score_embedding_basic():
    n = 50
    X = np.zeros((n, 1), dtype=np.float32)
    adata = ad.AnnData(X=X)
    adata.obsm["X_umap"] = np.random.RandomState(0).normal(size=(n, 2)).astype(np.float32)
    adata.obs["my_score"] = pd.Series(np.linspace(-1, 1, n))

    ax = sg.pl.score_embedding(adata, "my_score", basis="umap", title="ok")
    assert ax is not None


def test_score_embedding_missing_key_raises():
    n = 10
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_umap"] = np.random.RandomState(0).normal(size=(n, 2)).astype(np.float32)
    try:
        sg.pl.score_embedding(adata, "nope", basis="umap")
        assert False, "expected KeyError"
    except KeyError:
        pass
