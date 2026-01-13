from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ._shift import shift
from ._align_vectors import align_vectors


def velocity_delta_alignment(
    adata,
    *,
    # velocity vectors
    velocity_key: Optional[str] = None,  # default auto
    # shift inputs
    rep_for_shift: str = "X_umap",
    condition_key: str = "condition",
    group1: Any = None,
    group0: Any = None,
    by: Optional[str] = None,
    sample_key: Optional[str] = None,
    # alignment outputs
    shift_level: str = "global",          # global | by | samples
    shift_index_key: Optional[str] = None,
    obs_key: str = "scgeo_vel_delta_align",
    store_key: str = "scgeo",
) -> None:
    """
    Convenience wrapper for scVelo/CellRank workflows:
      1) compute Δ via sg.tl.shift on rep_for_shift
      2) align velocity vectors to Δ using sg.tl.align_vectors

    velocity_key auto-detection order:
      - "velocity_umap"
      - "velocity_pca"
    """
    if velocity_key is None:
        if "velocity_umap" in adata.obsm:
            velocity_key = "velocity_umap"
        elif "velocity_pca" in adata.obsm:
            velocity_key = "velocity_pca"
        else:
            raise KeyError("Could not auto-detect velocity vectors in obsm. Expected 'velocity_umap' or 'velocity_pca'.")

    # Step 1: compute Δ (stored under adata.uns[store_key]["shift"])
    shift(
        adata,
        rep=rep_for_shift,
        condition_key=condition_key,
        group1=group1,
        group0=group0,
        by=by,
        sample_key=sample_key,
        store_key=store_key,
    )

    # Step 2: align velocity to Δ
    align_vectors(
        adata,
        vec_key=velocity_key,
        ref_from_shift=True,
        shift_store_key=store_key,
        shift_kind="shift",
        shift_level=shift_level,
        shift_index_key=shift_index_key,
        obs_key=obs_key,
        store_key=store_key,
    )
