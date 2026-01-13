from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .._utils import _as_2d_array, cosine


def _get_shift_delta(
    adata,
    shift_store_key: str,
    shift_kind: str,
    shift_level: str,
) -> np.ndarray:
    if shift_store_key not in adata.uns or shift_kind not in adata.uns[shift_store_key]:
        raise KeyError(f"adata.uns['{shift_store_key}']['{shift_kind}'] not found")
    obj = adata.uns[shift_store_key][shift_kind]
    if shift_level not in obj:
        raise KeyError(f"shift level '{shift_level}' not found under {shift_store_key}.{shift_kind}")
    delta = obj[shift_level].get("delta", None)
    if delta is None:
        raise ValueError("shift delta is None (did you run sg.tl.shift with both groups present?)")
    return np.asarray(delta, dtype=np.float32)


def align_vectors(
    adata,
    vec_key: str,
    ref_vec_key: Optional[str] = None,
    *,
    ref_from_shift: bool = False,
    shift_store_key: str = "scgeo",
    shift_kind: str = "shift",
    shift_level: str = "global",  # "global" | "by" | "samples"
    shift_index_key: Optional[str] = None,  # required for by/samples
    obs_key: str = "scgeo_align",
    store_key: str = "scgeo",
) -> None:
    """
    Cosine alignment between vectors.

    Primary vectors:
      - adata.obsm[vec_key]  (n_cells x d)

    Reference:
      - if ref_vec_key provided: adata.obsm[ref_vec_key]  (n_cells x d)
      - if ref_from_shift=True:
          shift_level="global": uses single delta vector (broadcast to all cells)
          shift_level="by": uses adata.uns[...]["by"][level]["delta"], level from obs[shift_index_key]
          shift_level="samples": uses adata.uns[...]["samples"][sample]["delta"], sample from obs[shift_index_key]

    Writes:
      - adata.obs[obs_key] : per-cell cosine
      - adata.uns[store_key]["align_vectors"] : params + summary
    """
    if vec_key not in adata.obsm:
        raise KeyError(f"obsm['{vec_key}'] not found")
    V = _as_2d_array(adata.obsm[vec_key])

    n, d = V.shape
    ref_mode = None

    if ref_from_shift:
        # global ref
        if shift_level == "global":
            delta = _get_shift_delta(adata, shift_store_key, shift_kind, "global")
            if delta.shape[0] != d:
                raise ValueError(f"dim mismatch: vec dim={d} but shift delta dim={delta.shape[0]}")
            R = delta.reshape(1, -1)  # broadcast
            ref_mode = "shift:global"

        elif shift_level in ("by", "samples"):
            if shift_index_key is None:
                raise ValueError("shift_index_key is required when shift_level is 'by' or 'samples'")
            if shift_index_key not in adata.obs:
                raise KeyError(f"obs key '{shift_index_key}' not found")

            idx_vals = adata.obs[shift_index_key].astype(str).values
            obj = adata.uns.get(shift_store_key, {}).get(shift_kind, {})
            level_dict = obj.get(shift_level, None)
            if level_dict is None:
                raise KeyError(f"adata.uns['{shift_store_key}']['{shift_kind}']['{shift_level}'] not found")

            # Build per-cell reference vectors
            R = np.zeros((n, d), dtype=np.float32)
            missing = 0
            for i in range(n):
                key = idx_vals[i]
                ent = level_dict.get(key, None)
                if ent is None or ent.get("delta", None) is None:
                    missing += 1
                    R[i, :] = np.nan
                else:
                    delta = np.asarray(ent["delta"], dtype=np.float32)
                    if delta.shape[0] != d:
                        raise ValueError(f"dim mismatch for {shift_level}='{key}': vec dim={d}, delta dim={delta.shape[0]}")
                    R[i, :] = delta
            if missing > 0:
                # ok: these will become nan cosines
                pass
            ref_mode = f"shift:{shift_level}:{shift_index_key}"

        else:
            raise ValueError("shift_level must be one of: global, by, samples")

    else:
        if ref_vec_key is None:
            raise ValueError("Provide ref_vec_key or set ref_from_shift=True")
        if ref_vec_key not in adata.obsm:
            raise KeyError(f"obsm['{ref_vec_key}'] not found")
        R = _as_2d_array(adata.obsm[ref_vec_key])
        if R.shape != V.shape:
            raise ValueError(f"shape mismatch: {vec_key}={V.shape}, {ref_vec_key}={R.shape}")
        ref_mode = f"obsm:{ref_vec_key}"

    # compute per-cell cosine
    out = np.empty(n, dtype=np.float32)
    if R.shape[0] == 1:
        r = R[0]
        for i in range(n):
            out[i] = np.float32(cosine(V[i], r))
    else:
        for i in range(n):
            if np.any(np.isnan(R[i])):
                out[i] = np.nan
            else:
                out[i] = np.float32(cosine(V[i], R[i]))

    adata.obs[obs_key] = out

    if store_key not in adata.uns:
        adata.uns[store_key] = {}
    adata.uns[store_key]["align_vectors"] = {
        "params": dict(
            vec_key=vec_key,
            ref_mode=ref_mode,
            shift_store_key=shift_store_key if ref_from_shift else None,
            shift_kind=shift_kind if ref_from_shift else None,
            shift_level=shift_level if ref_from_shift else None,
            shift_index_key=shift_index_key if ref_from_shift else None,
            obs_key=obs_key,
        ),
        "summary": dict(
            mean=float(np.nanmean(out)),
            median=float(np.nanmedian(out)),
            min=float(np.nanmin(out)),
            max=float(np.nanmax(out)),
        ),
    }
