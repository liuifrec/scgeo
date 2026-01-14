import numpy as np
import anndata as ad
import scgeo as sg


def test_density_overlap_plot_grid_runs():
    n = 50
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_umap"] = np.random.RandomState(0).normal(size=(n, 2)).astype(np.float32)
    adata.uns["scgeo"] = {}
    adata.uns["scgeo"]["density_overlap"] = {
        "grid": {"xmin": -3, "xmax": 3, "ymin": -3, "ymax": 3},
        "overlap": np.random.RandomState(1).rand(40, 40),
        "group0": "A",
        "group1": "B",
    }
    fig, ax = sg.pl.density_overlap_grid(adata, show=False)
    assert fig is not None and ax is not None
