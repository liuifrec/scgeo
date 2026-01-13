import numpy as np
import pandas as pd
import anndata as ad


def test_paga_composition_stats_gee_runs():
    rs = np.random.RandomState(2)
    n_per_sample = 80
    samples_A = [f"sA{i}" for i in range(3)]
    samples_B = [f"sB{i}" for i in range(3)]

    node = []
    cond = []
    samp = []

    # Make node "X" enriched in condition B
    for s in samples_A:
        # mostly node Y
        node += list(rs.choice(["X", "Y"], size=n_per_sample, p=[0.2, 0.8]))
        cond += ["A"] * n_per_sample
        samp += [s] * n_per_sample
    for s in samples_B:
        # more node X
        node += list(rs.choice(["X", "Y"], size=n_per_sample, p=[0.6, 0.4]))
        cond += ["B"] * n_per_sample
        samp += [s] * n_per_sample

    obs = pd.DataFrame(
        {"leiden": node, "condition": cond, "sample": samp},
        index=[f"c{i}" for i in range(len(node))],
    )
    adata = ad.AnnData(X=np.zeros((len(node), 1)), obs=obs)

    import scgeo as sg
    sg.tl.paga_composition_stats(
        adata,
        group_key="leiden",
        condition_key="condition",
        group0="A",
        group1="B",
        sample_key="sample",
        method="gee",
    )

    tbl = adata.uns["scgeo"]["paga_composition_stats"]["table"]
    assert tbl.shape[0] >= 2
    assert set(["node", "OR", "CI_low", "CI_high", "p"]).issubset(tbl.columns)
