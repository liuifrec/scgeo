from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from .._utils import _as_2d_array



def consensus_subspace(
    adata: AnnData,
    rep: str = "X_pca",
    condition_key: str = "condition",
    group0: str | None = None,
    group1: str | None = None,
    sample_key: str | None = None,
    n_components: int = 2,
    obs_key_prefix: str = "cs",
    store_key: str = "consensus_subspace",
    min_cells: int = 20,
    center: bool = False,
) -> None:
    """
    Compute consensus subspace directions from multiple delta vectors (sample-aware if sample_key is given).

    Deltas are computed as mean(rep)[group1] - mean(rep)[group0] within each sample (or globally if sample_key None).
    Consensus directions are the top right singular vectors of the delta matrix.

    Stores:
      adata.uns["scgeo"][store_key] with components, singular_values, deltas, metadata
      adata.obsm[f"X_{obs_key_prefix}"] as cell projections onto consensus directions
      adata.obs[f"{obs_key_prefix}_score"] as per-cell magnitude in that subspace
    """
    if rep == "X":
        X = _as_2d_array(adata.X)
    else:
        X = _as_2d_array(adata.obsm[rep])
    # shape (n_cells, d)
    obs = adata.obs

    if condition_key not in obs.columns:
        raise KeyError(f"condition_key '{condition_key}' not found in adata.obs")

    cond = obs[condition_key].astype(str).values
    uniq = sorted(pd.unique(cond))
    if group0 is None or group1 is None:
        group0 = uniq[0]
        group1 = uniq[-1]

    mask0 = cond == str(group0)
    mask1 = cond == str(group1)
    if mask0.sum() == 0 or mask1.sum() == 0:
        raise ValueError(f"Groups not found or empty: group0={group0} (n={mask0.sum()}), group1={group1} (n={mask1.sum()})")

    deltas = []
    delta_index = []

    def mean_or_none(m: np.ndarray) -> np.ndarray | None:
        if m.sum() < min_cells:
            return None
        return X[m].mean(axis=0)

    if sample_key is None:
        mu0 = mean_or_none(mask0)
        mu1 = mean_or_none(mask1)
        if mu0 is None or mu1 is None:
            raise ValueError("Not enough cells to compute global delta")
        deltas.append(mu1 - mu0)
        delta_index.append("global")
    else:
        if sample_key not in obs.columns:
            raise KeyError(f"sample_key '{sample_key}' not found in adata.obs")

        samples = obs[sample_key].astype(str).values
        for s in sorted(pd.unique(samples)):
            ms = samples == s
            mu0 = mean_or_none(ms & mask0)
            mu1 = mean_or_none(ms & mask1)
            if mu0 is None or mu1 is None:
                continue
            deltas.append(mu1 - mu0)
            delta_index.append(s)

        if len(deltas) == 0:
            raise ValueError(
                f"No valid per-sample deltas found (min_cells={min_cells}). "
                f"Try lowering min_cells or check sample_key/condition groups."
            )

    D = np.vstack(deltas)  # (n_deltas, d)

    # Optional: center delta matrix (rarely needed)
    if center:
        D = D - D.mean(axis=0, keepdims=True)

    # SVD: D = U S Vt, rows are deltas; columns of V are consensus directions
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    k = int(min(n_components, Vt.shape[0]))
    components = Vt[:k, :]  # (k, d)

    # Project cells onto consensus subspace
    Z = X @ components.T  # (n_cells, k)
    score = np.linalg.norm(Z, axis=1)

    # Store
    adata.obsm[f"X_{obs_key_prefix}"] = Z
    adata.obs[f"{obs_key_prefix}_score"] = score

    adata.uns.setdefault("scgeo", {})
    adata.uns["scgeo"][store_key] = {
        "rep": rep,
        "condition_key": condition_key,
        "group0": str(group0),
        "group1": str(group1),
        "sample_key": None if sample_key is None else str(sample_key),
        "min_cells": int(min_cells),
        "center": bool(center),
        "n_deltas": int(D.shape[0]),
        "delta_index": delta_index,
        "deltas": D,
        "components": components,
        "singular_values": S[:k],
    }
