from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .._utils import _as_2d_array, _mask_from_obs


def shift(
    adata,
    rep: str = "X_pca",
    condition_key: str = "condition",
    group1: Any = None,
    group0: Any = None,
    by: Optional[str] = None,
    sample_key: Optional[str] = None,
    store_key: str = "scgeo",
) -> None:
    """
    Compute mean shift vector Δ = μ1 - μ0 in representation space.

    Stores results in:
      adata.uns[store_key]["shift"]["global"]
      adata.uns[store_key]["shift"]["by"][<level>] (if by is provided)
      adata.uns[store_key]["shift"]["samples"] (if sample_key is provided)
    """
    if rep not in adata.obsm:
        raise KeyError(f"obsm['{rep}'] not found")
    X = _as_2d_array(adata.obsm[rep])

    if group1 is None or group0 is None:
        # try infer two groups if not provided
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
            return {"n1": n1, "n0": n0, "mu1": None, "mu0": None, "delta": None, "delta_norm": np.nan}
        mu1 = X[mask1].mean(axis=0)
        mu0 = X[mask0].mean(axis=0)
        delta = mu1 - mu0
        return {
            "n1": n1,
            "n0": n0,
            "mu1": mu1.astype(np.float32),
            "mu0": mu0.astype(np.float32),
            "delta": delta.astype(np.float32),
            "delta_norm": float(np.linalg.norm(delta)),
        }

    out: Dict[str, Any] = {
        "params": dict(
            rep=rep,
            condition_key=condition_key,
            group1=group1,
            group0=group0,
            by=by,
            sample_key=sample_key,
        ),
        "global": _calc(m1, m0),
    }

    # by-stratified
    if by is not None:
        if by not in adata.obs:
            raise KeyError(f"obs key '{by}' not found")
        out_by = {}
        for level in adata.obs[by].astype(str).unique():
            mm = (adata.obs[by].astype(str).values == level)
            out_by[level] = _calc(m1 & mm, m0 & mm)
        out["by"] = out_by

    # per-sample (replicate deltas)
    if sample_key is not None:
        if sample_key not in adata.obs:
            raise KeyError(f"obs key '{sample_key}' not found")
        samples = adata.obs[sample_key].astype(str).unique()
        sample_out = {}
        for s in samples:
            ms = (adata.obs[sample_key].astype(str).values == s)
            sample_out[s] = _calc(m1 & ms, m0 & ms)
        out["samples"] = sample_out

    if store_key not in adata.uns:
        adata.uns[store_key] = {}
    adata.uns[store_key].setdefault("shift", {})
    adata.uns[store_key]["shift"] = out
