from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .._utils import _as_2d_array


def mixscore(
    adata,
    label_key: str = "batch",
    rep: str = "X_pca",
    k: int = 50,
    use_connectivities: bool = True,
    store_key: str = "scgeo",
    obs_key: str = "scgeo_mixscore",
) -> None:
    """
    kNN label mixing score in [0, 1].
    For each cell, compute the fraction of its neighbors having a *different* label.
    Higher => more mixed (more overlap), lower => more separated.

    If use_connectivities and adata.obsp['connectivities'] exists, use that graph
    (take top-k neighbors by connectivity for each cell).
    Otherwise compute kNN in embedding space adata.obsm[rep].
    """
    if label_key not in adata.obs:
        raise KeyError(f"obs key '{label_key}' not found")

    labels = adata.obs[label_key].astype(str).values
    n = adata.n_obs
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in [1, n_obs-1], got k={k}, n_obs={n}")

    neigh = None

    # --- Graph mode (scanpy-native) ---
    if use_connectivities and ("connectivities" in getattr(adata, "obsp", {})):
        C = adata.obsp["connectivities"]
        # sparse preferred; will work with csr/csc
        # For each row, pick top-k neighbor indices (excluding self if present)
        neigh = []
        for i in range(n):
            row = C[i]
            if hasattr(row, "toarray"):
                row = row.toarray().ravel()
            else:
                row = np.asarray(row).ravel()

            row[i] = 0.0  # drop self
            idx = np.argpartition(row, -k)[-k:]
            # sort by weight descending for determinism
            idx = idx[np.argsort(row[idx])[::-1]]
            neigh.append(idx.astype(np.int32))

        neigh = np.stack(neigh, axis=0)

    # --- kNN mode ---
    else:
        if rep not in adata.obsm:
            raise KeyError(f"obsm['{rep}'] not found")
        X = _as_2d_array(adata.obsm[rep])

        try:
            from sklearn.neighbors import NearestNeighbors
        except Exception as e:
            raise ImportError("scikit-learn required for kNN mode: pip install scgeo[sklearn]") from e

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
        nn.fit(X)
        _, idx = nn.kneighbors(X, return_distance=True)
        neigh = idx[:, 1:]  # drop self

    # mixing = fraction of neighbors with different label
    mix = np.empty(n, dtype=np.float32)
    for i in range(n):
        nb = neigh[i]
        mix[i] = np.mean(labels[nb] != labels[i]).astype(np.float32)

    adata.obs[obs_key] = mix

    if store_key not in adata.uns:
        adata.uns[store_key] = {}
    adata.uns[store_key]["mixscore"] = {
        "params": dict(label_key=label_key, rep=rep, k=k, use_connectivities=use_connectivities),
        "summary": {
            "mean": float(np.nanmean(mix)),
            "median": float(np.nanmedian(mix)),
            "min": float(np.nanmin(mix)),
            "max": float(np.nanmax(mix)),
        },
    }
