import numpy as np
import pandas as pd
import anndata as ad


def test_mixscore_knn_mode():
    rs = np.random.RandomState(0)
    X = rs.normal(size=(40, 8)).astype(np.float32)
    obs = pd.DataFrame({"batch": ["b1"] * 20 + ["b2"] * 20})
    adata = ad.AnnData(X=np.zeros((40, 1)), obs=obs)
    adata.obsm["X_pca"] = X

    import scgeo as sg
    sg.tl.mixscore(adata, label_key="batch", rep="X_pca", k=5, use_connectivities=False)
    assert "scgeo_mixscore" in adata.obs
    assert adata.obs["scgeo_mixscore"].between(0, 1).all()
