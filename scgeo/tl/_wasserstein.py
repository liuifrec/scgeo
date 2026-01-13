from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .._utils import _as_2d_array, _mask_from_obs


def _sliced_wasserstein_1d(a: np.ndarray, b: np.ndarray, p: int = 2) -> float:
    """1D Wasserstein between sorted samples (equal weights)."""
    a = np.sort(a.astype(np.float64, copy=False))
    b = np.sort(b.astype(np.float64, copy=False))
    n = min(a.size, b.size)
    if n == 0:
        return np.nan
    a = a[:n]
    b = b[:n]
    if p == 1:
        return float(np.mean(np.abs(a - b)))
    return float((np.mean(np.abs(a - b) ** p)) ** (1.0 / p))


def _sliced_wasserstein(
    X0: np.ndarray,
    X1: np.ndarray,
    n_proj: int = 128,
    p: int = 2,
    seed: int = 0,
) -> float:
    """
    Sliced Wasserstein distance in R^d:
      average over random 1D projections of 1D Wasserstein.
    """
    if X0.shape[0] == 0 or X1.shape[0] == 0:
        return np.nan
    d = X0.shape[1]
    rs = np.random.RandomState(seed)
    # random directions on unit sphere
    dirs = rs.normal(size=(n_proj, d)).astype(np.float64)
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

    vals = []
    for u in dirs:
        a = X0 @ u
        b = X1 @ u
        vals.append(_sliced_wasserstein_1d(a, b, p=p))
    return float(np.nanmean(vals))


def wasserstein(
    adata,
    rep: str = "X_pca",
    condition_key: str = "condition",
    group1: Any = None,
    group0: Any = None,
    *,
    by: Optional[str] = None,
    n_proj: int = 128,
    p: int = 2,
    seed: int = 0,
    store_key: str = "scgeo",
) -> None:
    """
    Compute (sliced) Wasserstein distance between two conditions in embedding space.

    Stores:
      adata.uns[store_key]["wasserstein"]["global"]
      adata.uns[store_key]["wasserstein"]["by"][level] (if by provided)

    Notes:
      - Uses sliced Wasserstein by default (no extra deps).
      - Equal-weight empirical distributions; uses min(n0, n1) truncation per projection.
    """
    if rep not in adata.obsm:
        raise KeyError(f"obsm['{rep}'] not found")
    X = _as_2d_array(adata.obsm[rep])

    if group1 is None or group0 is None:
        vals = list(adata.obs[condition_key].unique())
        if len(vals) != 2:
            raise ValueError(
                f"Need group1/group0 or exactly 2 unique values in obs['{condition_key}'], got {vals}"
            )
        group0, group1 = vals[0], vals[1]

    m1 = _mask_from_obs(adata, condition_key, group1)
    m0 = _mask_from_obs(adata, condition_key, group0)

    def _calc(mask1, mask0) -> Dict[str, Any]:
        n1 = int(mask1.sum())
        n0 = int(mask0.sum())
        if n1 == 0 or n0 == 0:
            return {"n1": n1, "n0": n0, "swd": np.nan}
        X1 = X[mask1]
        X0 = X[mask0]
        swd = _sliced_wasserstein(X0, X1, n_proj=n_proj, p=p, seed=seed)
        return {"n1": n1, "n0": n0, "swd": float(swd)}

    out: Dict[str, Any] = {
        "params": dict(
            rep=rep,
            condition_key=condition_key,
            group1=group1,
            group0=group0,
            by=by,
            n_proj=n_proj,
            p=p,
            seed=seed,
            method="sliced",
        ),
        "global": _calc(m1, m0),
    }

    if by is not None:
        if by not in adata.obs:
            raise KeyError(f"obs key '{by}' not found")
        out_by = {}
        levels = adata.obs[by].astype(str).unique()
        mm_all = adata.obs[by].astype(str).values
        for level in levels:
            mm = (mm_all == level)
            out_by[level] = _calc(m1 & mm, m0 & mm)
        out["by"] = out_by

    if store_key not in adata.uns:
        adata.uns[store_key] = {}
    adata.uns[store_key]["wasserstein"] = out
