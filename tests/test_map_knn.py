import numpy as np
import pandas as pd
import anndata as ad


def test_map_knn_basic():
    rs = np.random.RandomState(0)
    Xr = rs.normal(size=(60, 6)).astype(np.float32)
    yr = np.array(["T"] * 30 + ["B"] * 30)
    ref = ad.AnnData(X=np.zeros((60, 1)), obs=pd.DataFrame({"cell_type": yr}))
    ref.obsm["X_pca"] = Xr

    Xq = Xr[:20] + rs.normal(scale=0.01, size=(20, 6)).astype(np.float32)
    q = ad.AnnData(X=np.zeros((20, 1)), obs=pd.DataFrame(index=[f"q{i}" for i in range(20)]))
    q.obsm["X_pca"] = Xq

    import scgeo as sg
    sg.tl.map_knn(ref, q, label_key="cell_type", rep="X_pca", k=5)

    assert "scgeo_label" in q.obs
    assert "scgeo_confidence" in q.obs
    assert "scgeo_entropy" in q.obs
    assert "scgeo_ood" in q.obs
