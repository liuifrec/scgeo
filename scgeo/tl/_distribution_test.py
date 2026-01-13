from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from .._utils import _as_2d_array, _mask_from_obs

def _energy_distance(X0: np.ndarray, X1: np.ndarray) -> float:
    # energy distance: 2E||X-Y|| - E||X-X'|| - E||Y-Y'||
    # use random subset if huge later; keep exact for MVP
    def pdist(A, B):
        AA = (A**2).sum(1)[:, None]
        BB = (B**2).sum(1)[None, :]
        D2 = np.maximum(AA + BB - 2 * A @ B.T, 0.0)
        return np.sqrt(D2 + 1e-12)

    D01 = pdist(X0, X1).mean()
    D00 = pdist(X0, X0).mean()
    D11 = pdist(X1, X1).mean()
    return float(2 * D01 - D00 - D11)

def distribution_test(
    adata,
    rep: str = "X_pca",
    condition_key: str = "condition",
    group1: Any = None,
    group0: Any = None,
    *,
    sample_key: Optional[str] = None,
    by: Optional[str] = None,
    method: str = "energy",          # energy (MVP). MMD later.
    n_perm: int = 500,
    seed: int = 0,
    store_key: str = "scgeo",
) -> None:
    """
    Distribution difference test with sample-aware permutation.

    If sample_key provided:
      - compute observed statistic on all cells
      - permute condition labels at the SAMPLE level (swap sample assignments), recompute statistic
      - p-value = (1 + #{perm >= obs})/(1+n_perm)

    Stores:
      adata.uns[store_key]["distribution_test"]
    """
    if rep not in adata.obsm:
        raise KeyError(f"obsm['{rep}'] not found")
    X = _as_2d_array(adata.obsm[rep])

    if group1 is None or group0 is None:
        vals = list(adata.obs[condition_key].unique())
        if len(vals) != 2:
            raise ValueError("Need group0/group1 or exactly two condition values")
        group0, group1 = vals[0], vals[1]

    rs = np.random.RandomState(seed)

    def stat(mask1, mask0) -> float:
        X1 = X[mask1]
        X0 = X[mask0]
        if X1.shape[0] < 5 or X0.shape[0] < 5:
            return np.nan
        if method == "energy":
            return _energy_distance(X0, X1)
        raise ValueError("method must be 'energy' (MVP)")

    def _run(mask1, mask0, sub_obs) -> Dict[str, Any]:
        obs_stat = stat(mask1, mask0)
        if np.isnan(obs_stat):
            return {"stat": np.nan, "p_perm": np.nan, "n1": int(mask1.sum()), "n0": int(mask0.sum())}

        if sample_key is None:
            # naive permutation on cells (not recommended)
            y = sub_obs[condition_key].astype(str).values
            idx = np.arange(y.size)
            perm_stats = []
            for _ in range(n_perm):
                rs.shuffle(idx)
                y_perm = y[idx]
                m1p = (y_perm == str(group1))
                m0p = (y_perm == str(group0))
                perm_stats.append(stat(m1p, m0p))
        else:
            if sample_key not in sub_obs:
                raise KeyError(f"obs key '{sample_key}' not found")
            samp = sub_obs[sample_key].astype(str).values
            y = sub_obs[condition_key].astype(str).values

            # map sample -> condition (must be consistent; if not, still works but warning-worthy later)
            uniq_s = np.unique(samp)
            s2c = {}
            for s in uniq_s:
                vals = np.unique(y[samp == s])
                s2c[s] = vals[0]

            # permute conditions over samples (shuffle sample labels between groups)
            perm_stats = []
            for _ in range(n_perm):
                perm_s = uniq_s.copy()
                rs.shuffle(perm_s)
                s2c_perm = {s: s2c[perm_s[i]] for i, s in enumerate(uniq_s)}
                y_perm = np.array([s2c_perm[s] for s in samp], dtype=object)
                m1p = (y_perm == str(group1))
                m0p = (y_perm == str(group0))
                perm_stats.append(stat(m1p, m0p))

        perm_stats = np.array(perm_stats, dtype=np.float64)
        # higher stat => more different
        p = float((1.0 + np.sum(perm_stats >= obs_stat)) / (1.0 + np.sum(~np.isnan(perm_stats))))
        return {
            "stat": float(obs_stat),
            "p_perm": p,
            "n1": int(mask1.sum()),
            "n0": int(mask0.sum()),
            "n_perm": int(n_perm),
        }

    sub_obs = adata.obs

    m1 = _mask_from_obs(adata, condition_key, group1)
    m0 = _mask_from_obs(adata, condition_key, group0)

    out: Dict[str, Any] = {
        "params": dict(rep=rep, condition_key=condition_key, group0=group0, group1=group1, sample_key=sample_key, by=by, method=method, n_perm=n_perm, seed=seed),
        "global": _run(m1, m0, sub_obs),
    }

    if by is not None:
        if by not in adata.obs:
            raise KeyError(f"obs key '{by}' not found")
        mm_all = adata.obs[by].astype(str).values
        out_by = {}
        for level in np.unique(mm_all):
            mm = (mm_all == level)
            out_by[level] = _run(m1 & mm, m0 & mm, sub_obs.loc[mm])
        out["by"] = out_by

    adata.uns.setdefault(store_key, {})
    adata.uns[store_key]["distribution_test"] = out
