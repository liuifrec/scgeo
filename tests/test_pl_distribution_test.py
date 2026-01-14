import numpy as np
import anndata as ad
import scgeo as sg


def test_pl_distribution_test_global_runs():
    adata = ad.AnnData(X=np.zeros((10, 1), dtype=np.float32))
    adata.uns["scgeo"] = {
        "distribution_test": {
            "global": {"stat": 0.12, "p_perm": 0.03, "n0": 4, "n1": 6, "n_perm": 100},
            "by": {"x": {"stat": 0.3, "p_perm": 0.2, "n0": 2, "n1": 2, "n_perm": 100}},
        }
    }
    fig, axes = sg.pl.distribution_test(adata, level="global", show=False)
    assert fig is not None
    assert len(axes) == 1


def test_pl_distribution_test_by_runs():
    adata = ad.AnnData(X=np.zeros((10, 1), dtype=np.float32))
    adata.uns["scgeo"] = {
        "distribution_test": {
            "global": {"stat": 0.12, "p_perm": 0.03, "n0": 4, "n1": 6, "n_perm": 100},
            "by": {
                "A": {"stat": 0.10, "p_perm": 0.50, "n0": 10, "n1": 10, "n_perm": 100},
                "B": {"stat": 0.30, "p_perm": 0.01, "n0": 10, "n1": 10, "n_perm": 100},
                "C": {"stat": 0.20, "p_perm": 0.10, "n0": 10, "n1": 10, "n_perm": 100},
            },
        }
    }
    fig, axes = sg.pl.distribution_test(adata, level="by", top_k=2, highlight="strongest", show=False)
    assert fig is not None
    assert len(axes) == 1
