# scgeo/pp/_knn_graph.py
from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from .._utils import coo_to_csr

def build_query_to_ref_knn_edges(
    X_ref: np.ndarray,
    X_qry: np.ndarray,
    *,
    k: int = 30,
    metric: Literal["euclidean", "cosine"] = "cosine",
    mode: Literal["distance", "similarity"] = "similarity",
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (rows, cols, data) edges for a bipartite graph from query -> ref.

    rows: query row indices in [0, n_qry)
    cols: ref col indices in [0, n_ref)
    data: weights (float32)
    """
    from sklearn.neighbors import NearestNeighbors

    X_ref = np.asarray(X_ref, dtype=np.float32)
    X_qry = np.asarray(X_qry, dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X_ref)
    dists, idx = nn.kneighbors(X_qry, return_distance=True)  # (n_qry,k)

    rows = np.repeat(np.arange(X_qry.shape[0], dtype=np.int64), k)
    cols = idx.astype(np.int64).ravel()

    d = dists.astype(np.float32).ravel()

    if mode == "distance":
        data = d
    else:
        # similarity weight
        if sigma is None:
            # robust-ish default: median distance over all returned neighbors
            s = np.median(d) if d.size else 1.0
            sigma_eff = float(s) if s > 1e-8 else 1.0
        else:
            sigma_eff = float(sigma)

        data = np.exp(-(d ** 2) / (2.0 * sigma_eff ** 2)).astype(np.float32)

    return rows, cols, data


def build_block_connectivities_from_q2r(
    n_ref: int,
    n_qry: int,
    rows_q: np.ndarray,
    cols_r: np.ndarray,
    data: np.ndarray,
) :
    """
    Build a square CSR connectivities matrix for (ref + qry) nodes where
    edges go from query -> ref.

    Node indexing:
      ref: [0..n_ref)
      qry: [n_ref..n_ref+n_qry)
    """
    rows = rows_q + n_ref
    cols = cols_r
    shape = (n_ref + n_qry, n_ref + n_qry)
    return coo_to_csr(rows, cols, data, shape, dtype=np.float32, sum_duplicates=True)
