import numpy as np
import pandas as pd
import anndata as ad


def test_align_vectors_global_ref():
    rs = np.random.RandomState(0)
    V = rs.normal(size=(30, 4)).astype(np.float32)

    obs = pd.DataFrame({"cell_type": ["A"] * 15 + ["B"] * 15}, index=[f"c{i}" for i in range(30)])
    adata = ad.AnnData(X=np.zeros((30, 1)), obs=obs)
    adata.obsm["velocity_umap"] = V

    # fake shift delta
    adata.uns["scgeo"] = {"shift": {"global": {"delta": np.array([1, 0, 0, 0], dtype=np.float32)}}}

    import scgeo as sg
    sg.tl.align_vectors(
        adata,
        vec_key="velocity_umap",
        ref_from_shift=True,
        obs_key="align_v_delta",
    )

    assert "align_v_delta" in adata.obs
    assert "align_vectors" in adata.uns["scgeo"]


def test_align_vectors_pairwise():
    rs = np.random.RandomState(1)
    V1 = rs.normal(size=(20, 3)).astype(np.float32)
    V2 = V1.copy()

    obs = pd.DataFrame(index=[f"c{i}" for i in range(20)])
    adata = ad.AnnData(X=np.zeros((20, 1)), obs=obs)
    adata.obsm["v1"] = V1
    adata.obsm["v2"] = V2

    import scgeo as sg
    sg.tl.align_vectors(adata, vec_key="v1", ref_vec_key="v2", obs_key="align")

    # identical vectors -> cosine ~1
    assert float(np.nanmin(adata.obs["align"])) > 0.99
