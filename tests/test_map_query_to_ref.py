import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import scgeo as sg


def test_map_query_to_ref_graph_vote_basic():
    n_ref = 40
    n_qry = 20
    n = n_ref + n_qry

    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obs["batch"] = ["ref"] * n_ref + ["qry"] * n_qry
    adata.obs["label"] = ["A"] * 20 + ["B"] * 20 + [None] * n_qry

    # Build a simple connectivities matrix:
    # each query cell connects strongly to either first 20 (A) or next 20 (B)
    rows = []
    cols = []
    data = []
    for i in range(n_ref, n):
        if (i - n_ref) < (n_qry // 2):
            nbrs = list(range(0, 10))  # A
        else:
            nbrs = list(range(20, 30))  # B
        for j in nbrs:
            rows.append(i)
            cols.append(j)
            data.append(0.2)
    C = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    adata.obsp["connectivities"] = C

    sg.tl.map_query_to_ref(
        adata,
        ref_key="batch",
        ref_value="ref",
        label_key="label",
        graph_key="connectivities",
    )

    preds = adata.obs.loc[adata.obs["batch"] == "qry", "scgeo_pred"].astype(str).tolist()
    assert preds.count("A") > 0
    assert preds.count("B") > 0

    conf = adata.obs.loc[adata.obs["batch"] == "qry", "scgeo_conf"].to_numpy()
    ood = adata.obs.loc[adata.obs["batch"] == "qry", "scgeo_ood"].to_numpy()
    assert np.all(np.isfinite(conf))
    assert np.all(np.isfinite(ood))
