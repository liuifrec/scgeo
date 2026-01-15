import numpy as np
import anndata as ad
import scgeo as sg


def test_map_query_to_ref_pool_runs():
    rng = np.random.RandomState(0)

    # query adata with embedding
    n = 40
    Xq = rng.normal(size=(n, 8)).astype(np.float32)
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_emb"] = Xq

    # build pool
    Xref = rng.normal(size=(60, 8)).astype(np.float32)
    labels = np.array(["A"] * 30 + ["B"] * 30, dtype=object)
    pool = sg.pp.build_reference_pool(Xref, {"label": labels}, label_key="label", n_neighbors=15)

    sg.tl.map_query_to_ref_pool(adata, pool, rep="X_emb", k=10, return_probs=True)

    assert "scgeo_pred" in adata.obs
    assert "scgeo_conf" in adata.obs
    assert "scgeo_ood" in adata.obs
    assert "X_map_probs" in adata.obsm
    assert "scgeo" in adata.uns
    assert "map_query_to_ref" in adata.uns["scgeo"]
