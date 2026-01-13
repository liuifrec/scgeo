import os
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.environ.get("SCGEO_INTEGRATION") != "1", reason="integration test (set SCGEO_INTEGRATION=1)")
def test_pbmc3k_tutorial_flow():
    import scanpy as sc
    import scgeo as sg

    adata = sc.datasets.pbmc3k()  # cached under ~/.cache/scanpy after first run
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1500)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)

    # Pretend "batch" for mixscore (split by parity)
    adata.obs["batch"] = ["b1" if i % 2 == 0 else "b2" for i in range(adata.n_obs)]

    # mixscore should use connectivities
    sg.tl.mixscore(adata, label_key="batch", k=25, use_connectivities=True)
    assert "scgeo_mixscore" in adata.obs

    # mapping test: use half as ref with fake labels
    adata.obs["cell_type"] = ["T" if i % 3 == 0 else "B" for i in range(adata.n_obs)]
    ref = adata[: adata.n_obs // 2].copy()
    qry = adata[adata.n_obs // 2 :].copy()
    sg.tl.map_knn(ref, qry, label_key="cell_type", rep="X_pca", k=15)
    assert "scgeo_label" in qry.obs
