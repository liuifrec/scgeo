import numpy as np
import pandas as pd
import anndata as ad


def test_distribution_test_sample_perm_runs():
    rs = np.random.RandomState(1)
    n_per_sample = 60
    d = 4

    # 6 samples: 3 in A, 3 in B
    samples_A = [f"sA{i}" for i in range(3)]
    samples_B = [f"sB{i}" for i in range(3)]

    X_list = []
    cond = []
    samp = []
    for s in samples_A:
        X_list.append(rs.normal(loc=0.0, scale=1.0, size=(n_per_sample, d)))
        cond += ["A"] * n_per_sample
        samp += [s] * n_per_sample
    for s in samples_B:
        X_list.append(rs.normal(loc=0.8, scale=1.0, size=(n_per_sample, d)))
        cond += ["B"] * n_per_sample
        samp += [s] * n_per_sample

    X = np.vstack(X_list).astype(np.float32)

    obs = pd.DataFrame(
        {"condition": cond, "sample": samp},
        index=[f"c{i}" for i in range(X.shape[0])],
    )
    adata = ad.AnnData(X=np.zeros((X.shape[0], 1)), obs=obs)
    adata.obsm["X_pca"] = X

    import scgeo as sg
    sg.tl.distribution_test(
        adata,
        rep="X_pca",
        condition_key="condition",
        group0="A",
        group1="B",
        sample_key="sample",
        method="energy",
        n_perm=100,
        seed=0,
    )

    out = adata.uns["scgeo"]["distribution_test"]["global"]
    assert "stat" in out and "p_perm" in out
    assert 0.0 <= out["p_perm"] <= 1.0
