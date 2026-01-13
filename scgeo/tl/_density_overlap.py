from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .._utils import _as_2d_array, _mask_from_obs

def _knn_density(X_ref: np.ndarray, X_eval: np.ndarray, k: int = 30, eps: float = 1e-12) -> np.ndarray:
    """
    kNN density estimate: p(x) ∝ 1 / r_k(x)^d
    Uses distances to k-th NN in reference set.
    """
    d = X_ref.shape[1]
    k_eff = min(k, max(2, X_ref.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(X_ref)
    dist, _ = nn.kneighbors(X_eval, return_distance=True)
    rk = dist[:, -1]
    rk = np.maximum(rk, eps)
    # up to multiplicative constant, ok because we normalize later
    return 1.0 / (rk ** d)

def density_overlap(
    adata,
    rep: str = "X_umap",
    condition_key: str = "condition",
    group1: Any = None,
    group0: Any = None,
    *,
    by: Optional[str] = None,
    k: int = 30,
    eval_on: str = "union",   # union | group0 | group1
    store_key: str = "scgeo",
) -> None:
    """
    Compute density overlap between two conditions on an embedding using kNN density estimates.
    Outputs Bhattacharyya coefficient (BC) and Hellinger distance (H).

    BC = ∫ sqrt(p q) dx   (in [0,1], higher = more overlap)
    H  = sqrt(1 - BC)     (in [0,1], higher = more separated)
    """
    if rep not in adata.obsm:
        raise KeyError(f"obsm['{rep}'] not found")
    X = _as_2d_array(adata.obsm[rep])

    if group1 is None or group0 is None:
        vals = list(adata.obs[condition_key].unique())
        if len(vals) != 2:
            raise ValueError(f"Need group1/group0 or exactly 2 unique values in obs['{condition_key}']")
        group0, group1 = vals[0], vals[1]

    m1 = _mask_from_obs(adata, condition_key, group1)
    m0 = _mask_from_obs(adata, condition_key, group0)

    def _calc(mask1, mask0) -> Dict[str, float]:
        X1 = X[mask1]
        X0 = X[mask0]
        if X1.shape[0] < 5 or X0.shape[0] < 5:
            return {"n1": int(X1.shape[0]), "n0": int(X0.shape[0]), "bc": np.nan, "hellinger": np.nan}

        if eval_on == "group0":
            X_eval = X0
        elif eval_on == "group1":
            X_eval = X1
        else:
            X_eval = np.vstack([X0, X1])

        p = _knn_density(X0, X_eval, k=k).astype(np.float64)
        q = _knn_density(X1, X_eval, k=k).astype(np.float64)

        # normalize to discrete probability over evaluation points
        p /= (p.sum() + 1e-12)
        q /= (q.sum() + 1e-12)

        bc = float(np.sum(np.sqrt(p * q)))
        bc = float(np.clip(bc, 0.0, 1.0))
        hell = float(np.sqrt(max(0.0, 1.0 - bc)))
        return {"n1": int(X1.shape[0]), "n0": int(X0.shape[0]), "bc": bc, "hellinger": hell}

    out: Dict[str, Any] = {
        "params": dict(rep=rep, condition_key=condition_key, group0=group0, group1=group1, by=by, k=k, eval_on=eval_on),
        "global": _calc(m1, m0),
    }

    if by is not None:
        if by not in adata.obs:
            raise KeyError(f"obs key '{by}' not found")
        mm_all = adata.obs[by].astype(str).values
        out_by = {}
        for level in np.unique(mm_all):
            mm = (mm_all == level)
            out_by[level] = _calc(m1 & mm, m0 & mm)
        out["by"] = out_by

    adata.uns.setdefault(store_key, {})
    adata.uns[store_key]["density_overlap"] = out
