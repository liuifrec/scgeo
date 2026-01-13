import numpy as np
import pandas as pd
import anndata as ad


def test_consensus_subspace_recovers_direction():
    rs = np.random.RandomState(0)
    n_per_sample = 80
    d = 5
    true = np.zeros(d)
    true[0] = 1.0

    samples = [f"s{i}" for i in range(6)]
    Xs = []
    obs_rows = []

    # Each sample has BOTH conditions (A/B), with a consistent direction shift
    for s in samples:
        # sample-specific baseline
        base = rs.normal(0, 0.2, size=(1, d))

        # A cells
        XA = rs.normal(0, 1, size=(n_per_sample, d)) + base
        # B cells shifted along true direction
        XB = rs.normal(0, 1, size=(n_per_sample, d)) + base + 0.8 * true

        Xs.append(XA)
        Xs.append(XB)

        obs_rows += [{"condition": "A", "sample": s} for _ in range(n_per_sample)]
        obs_rows += [{"condition": "B", "sample": s} for _ in range(n_per_sample)]

    X = np.vstack(Xs).astype(np.float32)
    obs = pd.DataFrame(obs_rows)

    adata = ad.AnnData(X=np.zeros((X.shape[0], 1)), obs=obs)
    adata.obsm["X_pca"] = X

    import scgeo as sg
    sg.tl.consensus_subspace(
        adata,
        rep="X_pca",
        condition_key="condition",
        group0="A",
        group1="B",
        sample_key="sample",
        n_components=1,
        obs_key_prefix="cs",
        min_cells=10,
    )

    comp = adata.uns["scgeo"]["consensus_subspace"]["components"][0]
    cos = abs(np.dot(comp, true) / (np.linalg.norm(comp) * np.linalg.norm(true)))
    assert cos > 0.9
    assert "cs_score" in adata.obs
    assert "X_cs" in adata.obsm
