from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._utils import _as_2d_array, cosine


def _normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.where(nrm < eps, np.nan, nrm)
    return X / nrm


def _vectors_from_obsm(adata, key: str) -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"obsm['{key}'] not found")
    return _as_2d_array(adata.obsm[key])


def _vectors_from_shift(
    adata,
    *,
    store_key: str = "scgeo",
    kind: str = "shift",
    level: str = "global",     # global | by | samples
    index_key: Optional[str] = None,  # required for by/samples
) -> np.ndarray:
    if store_key not in adata.uns or kind not in adata.uns[store_key]:
        raise KeyError(f"adata.uns['{store_key}']['{kind}'] not found")

    obj = adata.uns[store_key][kind]

    # global broadcast
    if level == "global":
        delta = obj["global"].get("delta", None)
        if delta is None:
            raise ValueError("shift global delta is None; run sg.tl.shift first")
        delta = np.asarray(delta, dtype=np.float32)
        return np.repeat(delta.reshape(1, -1), adata.n_obs, axis=0)

    # by / samples per-cell
    if level in ("by", "samples"):
        if index_key is None:
            raise ValueError("index_key is required for level='by' or 'samples'")
        if index_key not in adata.obs:
            raise KeyError(f"obs key '{index_key}' not found")
        mapping = obj.get(level, None)
        if mapping is None:
            raise KeyError(f"shift level '{level}' not found in adata.uns['{store_key}']['{kind}']")

        keys = adata.obs[index_key].astype(str).values
        # determine dim from first available delta
        dim = None
        for v in mapping.values():
            if v.get("delta", None) is not None:
                dim = len(v["delta"])
                break
        if dim is None:
            raise ValueError(f"No non-null deltas found under shift level '{level}'")

        out = np.full((adata.n_obs, dim), np.nan, dtype=np.float32)
        for i, k in enumerate(keys):
            ent = mapping.get(k, None)
            if ent is None or ent.get("delta", None) is None:
                continue
            out[i] = np.asarray(ent["delta"], dtype=np.float32)
        return out

    raise ValueError("level must be one of: global, by, samples")


def projection_disagreement(
    adata,
    sources: Sequence[Dict[str, Any]],
    obs_key: str = "scgeo_disagree",
    store_key: str = "scgeo",
) -> None:
    """
    Compute per-cell projection disagreement among multiple vector sources.

    sources: list of dicts describing vector sources, e.g.
      {"type": "obsm", "key": "velocity_umap", "name": "velocity"}
      {"type": "shift", "level": "global", "name": "delta_global"}
      {"type": "shift", "level": "by", "index_key": "cell_type", "name": "delta_by_ct"}

    Output:
      adata.obs[obs_key] = 1 - mean_pairwise_cosine (nan-safe)
      adata.uns[store_key]["projection_disagreement"] contains params + summary
    """
    if len(sources) < 2:
        raise ValueError("Need at least 2 sources to compute disagreement")

    vecs: List[np.ndarray] = []
    names: List[str] = []

    for s in sources:
        stype = s.get("type", None)
        name = s.get("name", None) or stype

        if stype == "obsm":
            X = _vectors_from_obsm(adata, s["key"])
        elif stype == "shift":
            X = _vectors_from_shift(
                adata,
                store_key=s.get("store_key", "scgeo"),
                kind=s.get("kind", "shift"),
                level=s.get("level", "global"),
                index_key=s.get("index_key", None),
            )
        else:
            raise ValueError(f"Unknown source type: {stype}")

        vecs.append(X.astype(np.float32, copy=False))
        names.append(name)

    # shape checks
    d = vecs[0].shape[1]
    for X in vecs:
        if X.shape[0] != adata.n_obs:
            raise ValueError("All sources must have n_obs rows")
        if X.shape[1] != d:
            raise ValueError("All sources must have same vector dimension")

    # normalize
    vecs = [_normalize_rows(X) for X in vecs]

    m = len(vecs)
    disagree = np.full(adata.n_obs, np.nan, dtype=np.float32)

    for i in range(adata.n_obs):
        # pairwise cosine among available vectors
        cs = []
        for a in range(m):
            va = vecs[a][i]
            if np.any(np.isnan(va)):
                continue
            for b in range(a + 1, m):
                vb = vecs[b][i]
                if np.any(np.isnan(vb)):
                    continue
                cs.append(cosine(va, vb))
        if len(cs) == 0:
            disagree[i] = np.nan
        else:
            disagree[i] = np.float32(1.0 - float(np.nanmean(cs)))

    adata.obs[obs_key] = disagree

    if store_key not in adata.uns:
        adata.uns[store_key] = {}
    adata.uns[store_key]["projection_disagreement"] = {
        "params": {"sources": sources, "obs_key": obs_key},
        "summary": {
            "mean": float(np.nanmean(disagree)),
            "median": float(np.nanmedian(disagree)),
            "min": float(np.nanmin(disagree)),
            "max": float(np.nanmax(disagree)),
        },
    }
