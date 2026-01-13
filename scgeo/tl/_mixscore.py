from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .._utils import _as_2d_array


def _topk_from_csr_row(indices, data, k: int) -> np.ndarray:
    """Return indices of top-k entries by weight (descending) for one CSR row."""
    if indices.size == 0:
        return indices
    if indices.size <= k:
        # sort all by weight desc
        order = np.argsort(data)[::-1]
        return indices[order]
    # partial select then sort
    sel = np.argpartition(data, -k)[-k:]
    sel = sel[np.argsort(data[sel])[::-1]]
    return indices[sel]


def mixscore(
    adata,
    label_key: str = "batch",
    rep: str = "X_pca",
    k: int = 50,
    use_connectivities: bool = True,
    connectivities_key: str = "connectivities",
    obs_key: str = "scgeo_mixscore",
    store_key: str = "scgeo",
) -> None:
    """
    kNN label mixing score in [0,1]:
      per cell = fraction of neighbors with a different label.

    Preference order:
      1) adata.obsp[connectivities_key] if present and use_connectivities=True (scanpy/bbknn native)
      2) kNN on adata.obsm[rep] (requires scikit-learn)

    Writes:
      adata.obs[obs_key]
      adata.uns[store_key]["mixscore"]
    """
    if label_key not in adata.obs:
        raise KeyError(f"obs key '{label_key}' not found")

    labels = adata.obs[label_key].astype(str).values
    n = adata.n_obs
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in [1, n_obs-1], got k={k}, n_obs={n}")

    neigh = None

    # --- graph mode ---
    if use_connectivities and (connectivities_key in getattr(adata, "obsp", {})):
        C = adata.obsp[connectivities_key]
        # require CSR-like access
        if not hasattr(C, "indptr") or not hasattr(C, "indices") or not hasattr(C, "data"):
            # fallback to dense-ish behavior
            C = C.tocsr()

        C = C.tocsr()
        indptr = C.indptr
        indices = C.indices
        data = C.data

        neigh = np.empty((n, k), dtype=np.int32)
        for i in range(n):
            start, end = indptr[i], indptr[i + 1]
            idx = indices[start:end]
            w = data[start:end]

            # drop self if present
            if idx.size > 0:
                m = idx != i
                idx = idx[m]
                w = w[m]

            top = _topk_from_csr_row(idx.astype(np.int32), w.astype(np.float32), k)
            # if fewer than k neighbors exist, pad by repeating last (rare but safe)
            if top.size == 0:
                # no neighbors: self-only fallback (mixscore = 0)
                top = np.array([i], dtype=np.int32)
            if top.size < k:
                top = np.pad(top, (0, k - top.size), mode="edge")
            neigh[i] = top[:k]

    # --- embedding kNN mode ---
    else:
        if rep not in adata.obsm:
            raise KeyError(f"obsm['{rep}'] not found")
        X = _as_2d_array(adata.obsm[rep])

        try:
            from sklearn.neighbors import NearestNeighbors
        except Exception as e:
            raise ImportError("scikit-learn required for kNN mode: pip install scgeo[sklearn]") from e

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(X)
        _, idx = nn.kneighbors(X, return_distance=True)
        neigh = idx[:, 1:].astype(np.int32)

    mix = np.empty(n, dtype=np.float32)
    for i in range(n):
        nb = neigh[i]
        mix[i] = float(np.mean(labels[nb] != labels[i]))

    adata.obs[obs_key] = mix

    if store_key not in adata.uns:
        adata.uns[store_key] = {}
    adata.uns[store_key]["mixscore"] = {
        "params": dict(
            label_key=label_key,
            rep=rep,
            k=k,
            use_connectivities=use_connectivities,
            connectivities_key=connectivities_key,
            obs_key=obs_key,
        ),
        "summary": {
            "mean": float(np.nanmean(mix)),
            "median": float(np.nanmedian(mix)),
            "min": float(np.nanmin(mix)),
            "max": float(np.nanmax(mix)),
        },
    }
