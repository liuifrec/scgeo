import numpy as np
import pandas as pd
import anndata as ad


def test_shift_basic():
    X = np.random.RandomState(0).normal(size=(20, 5)).astype(np.float32)
    obs = pd.DataFrame({
        "condition": ["A"] * 10 + ["B"] * 10,
        "cell_type": ["T"] * 5 + ["B"] * 5 + ["T"] * 5 + ["B"] * 5,
        "donor": ["d1"] * 10 + ["d2"] * 10,
    })
    adata = ad.AnnData(X=np.zeros((20, 1)), obs=obs)
    adata.obsm["X_emb"] = X

    import scgeo as sg
    sg.tl.shift(adata, rep="X_emb", condition_key="condition", group0="A", group1="B", by="cell_type", sample_key="donor")

    assert "scgeo" in adata.uns
    assert "shift" in adata.uns["scgeo"]
    tab = sg.get.table(adata, level="by")
    assert set(tab.columns) >= {"name", "n1", "n0", "delta_norm"}
    assert len(tab) == 2
