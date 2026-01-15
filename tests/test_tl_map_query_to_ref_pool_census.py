import numpy as np
import anndata as ad
import scgeo as sg

def test_map_query_to_ref_pool_census_pool_mode_runs():
    rng = np.random.RandomState(0)

    # query data
    n_q = 40
    d = 8
    Xq = rng.normal(size=(n_q, d)).astype(np.float32)
    adata_q = ad.AnnData(X=np.zeros((n_q, 1), dtype=np.float32))
    adata_q.obsm["X_emb"] = Xq

    # reference pool (no census)
    n_ref = 80
    Xref = rng.normal(size=(n_ref, d)).astype(np.float32)
    labels = np.array(["A"] * (n_ref // 2) + ["B"] * (n_ref - n_ref // 2), dtype=object)

    # give joinids so census-knn path remains compatible later
    joinids = np.arange(1000, 1000 + n_ref, dtype=np.int64)
    pool = sg.pp.build_reference_pool(
        Xref,
        {"label": labels},
        label_key="label",
        n_neighbors=15,
        meta={"source": "test"},
        joinids=joinids,
    )

    sg.tl.map_query_to_ref_pool_census(
        adata_q,
        pool=pool,  # pool-mode (no census)
        rep="X_emb",
        label_key="label",
        return_probs=True,
    )

    assert "scgeo_pred" in adata_q.obs
    assert "scgeo_conf" in adata_q.obs
    assert "scgeo_ood" in adata_q.obs
    assert "X_map_probs" in adata_q.obsm
    assert "scgeo" in adata_q.uns
    assert "map_query_to_ref" in adata_q.uns["scgeo"]
