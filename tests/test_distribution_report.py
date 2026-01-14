import numpy as np
import pandas as pd
import anndata as ad
import scgeo as sg


def test_distribution_report_runs():
    n = 120
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_umap"] = np.random.RandomState(0).normal(size=(n, 2)).astype(np.float32)
    adata.obs["condition"] = pd.Series(["A"] * 60 + ["B"] * 60, index=adata.obs.index)
    adata.obs["cs_score"] = pd.Series(np.random.RandomState(1).rand(n), index=adata.obs.index)

    adata.uns["scgeo"] = {
        "density_overlap": {
            "global": {"bc": 0.6, "hellinger": 0.4, "n0": 60, "n1": 60},
            "by": {"0": {"bc": 0.2, "hellinger": 0.8, "n0": 10, "n1": 10}},
        },
        "distribution_test": {
            "global": {"stat": 0.12, "p_perm": 0.03, "n0": 60, "n1": 60, "n_perm": 100},
            "by": {"0": {"stat": 0.4, "p_perm": 0.01, "n0": 10, "n1": 10, "n_perm": 100}},
        },
    }

    fig, axes = sg.pl.distribution_report(adata, score_key="cs_score", show=False)
    assert fig is not None
    assert axes.shape == (2, 2)
