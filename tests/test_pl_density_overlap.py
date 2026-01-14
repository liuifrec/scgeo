import numpy as np
import anndata as ad
import scgeo as sg


def test_pl_density_overlap_global_runs():
    adata = ad.AnnData(X=np.zeros((10, 1), dtype=np.float32))
    adata.uns["scgeo"] = {
        "density_overlap": {
            "global": {"bc": 0.5, "hellinger": 0.3, "n0": 4, "n1": 6},
            "by": {"x": {"bc": 0.2, "hellinger": 0.8, "n0": 2, "n1": 2}},
        }
    }
    fig, axes = sg.pl.density_overlap(adata, level="global", show=False)
    assert fig is not None
    assert len(axes) == 1


def test_pl_density_overlap_by_runs():
    adata = ad.AnnData(X=np.zeros((10, 1), dtype=np.float32))
    adata.uns["scgeo"] = {
        "density_overlap": {
            "global": {"bc": 0.5, "hellinger": 0.3, "n0": 4, "n1": 6},
            "by": {
                "A": {"bc": 0.9, "hellinger": 0.1, "n0": 10, "n1": 10},
                "B": {"bc": 0.2, "hellinger": 0.8, "n0": 10, "n1": 10},
                "C": {"bc": 0.4, "hellinger": 0.6, "n0": 10, "n1": 10},
            },
        }
    }
    fig, axes = sg.pl.density_overlap(adata, level="by", top_k=2, highlight="worst", show=False)
    assert fig is not None
    assert len(axes) == 2
