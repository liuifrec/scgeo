import numpy as np
import pandas as pd
import anndata as ad
import scgeo as sg


def test_distribution_test_volcano_runs():
    adata = ad.AnnData(X=np.zeros((10, 1), dtype=np.float32))
    adata.uns["scgeo"] = {}
    adata.uns["scgeo"]["distribution_test"] = pd.DataFrame(
        {
            "group": ["a", "b", "c"],
            "effect": [0.1, 0.5, -0.2],
            "p_adj": [0.2, 1e-6, 0.03],
        }
    )
    fig, ax, df = sg.pl.distribution_test_volcano(adata, show=False)
    assert fig is not None and ax is not None
    assert df.shape[0] == 3
