from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .._utils import _as_2d_array, cosine


def align_vectors(
    adata,
    vec_key: str,
    ref_vec_key: Optional[str] = None,
    ref_from_shift: bool = False,
    shift_store_key: str = "scgeo",
    shift_kind: str = "shift",
    shift_level: str = "global",
    by: Optional[str] = None,
    obs_key: str = "scgeo_align",
    store_key: str = "scgeo",
) -> None:
    """
    Cosine alignment between vectors.

    Primary vector field: adata.obsm[vec_key]  (n_cells x d)

    Reference vector:
      - if ref_vec_key is provided: adata.obsm[ref_vec_key]  (n_cells x d)
      - if ref_from_shift=True: use delta from adata.uns[shift_store_key][shift_kind][shift_level]["delta"] (d,)
        (broadcast to all cells)

    Writes:
      - adata.obs[obs_key] : per-cell cosine
      - adata.uns[store_key]["align_vectors"] : params + optional by-summary
    """
    if vec_key not in adata.obsm:
        raise KeyError(f"obsm['{vec_key}'] not found")
    V = _as_2d_array(adata.obsm[vec_key])

    # build reference
    if ref_from_shift:
        if shift_store_key not in adata.uns or shift_kind not in adata.uns[shift_store_key]:
            raise KeyError(f"adata.uns['{shift_store_key}']['{shift_kind}'] not found")
        obj = adata.uns[shift_store_key][shift_kind]
        if shift_level not in obj:
            raise KeyError(f"shift level '{shift_level}' not found under {shift_store_key}.{shift_kind}")
        delta = obj[shift_level].get("delta", None)
        if delta is None:
            raise ValueError("shift delta is None (did you run sg.tl.shift with both groups present?)")
        R = np.asarray(delta, dtype=np.float32).reshape(1, -1)
        if R.shape[1] != V.shape[1]:
            raise ValueError(f"dim mismatch: vec dim={V.shape[1]} but shift delta dim={R.shape[1]}")
        # broadcast later
        ref_mode = f"shift:{shift_level}"
    else:
        if ref_vec_key is None:
            raise ValueError("Provide ref_vec_key or set ref_from_shift=True")
        if ref_vec_key not in adata.obsm:
            raise KeyError(f"obsm['{ref_vec_key}'] not found")
        R = _as_2d_array(adata.obsm[ref_vec_key])
        if R.shape != V.shape:
            raise ValueError(f"shape mismatch: {vec_key}={V.shape}, {ref_vec_key}={R.shape}")
        ref_mode = f"obsm:{ref_vec_key}"

    # compute cosine per cell
    out = np.empty(adata.n_obs, dtype=np.float32)
    if R.shape[0] == 1:  # global reference
        r = R[0]
        for i in range(adata.n_obs):
            out[i] = np.float32(cosine(V[i], r))
    else:
        for i in range(adata.n_obs):
            out[i] = np.float32(cosine(V[i], R[i]))

    adata.obs[obs_key] = out

    # store summary
    if store_key not in adata.uns:
        adata.uns[store_key] = {}
    payload: Dict[str, Any] = {
        "params": dict(
            vec_key=vec_key,
            ref_mode=ref_mode,
            by=by,
            obs_key=obs_key,
        ),
        "summary": dict(
            mean=float(np.nanmean(out)),
            median=float(np.nanmedian(out)),
            min=float(np.nanmin(out)),
            max=float(np.nanmax(out)),
        ),
    }

    if by is not None:
        if by not in adata.obs:
            raise KeyError(f"obs key '{by}' not found")
        grp = adata.obs[by].astype(str).values
        by_sum = {}
        for g in np.unique(grp):
            m = grp == g
            by_sum[g] = {
                "n": int(m.sum()),
                "mean": float(np.nanmean(out[m])),
                "median": float(np.nanmedian(out[m])),
            }
        payload["by"] = by_sum

    adata.uns[store_key]["align_vectors"] = payload
